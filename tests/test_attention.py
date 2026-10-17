"""nn/attention.py のテスト。

PyTorch 組み込みの実装（F.scaled_dot_product_attention / nn.MultiheadAttention）と数値を比べて、
自前実装が正しいかを確かめる。組み込みの実装を使うのはテストの中だけ（src では使わない）。

    uv run pytest tests/test_attention.py -v
"""

import torch
from torch import nn
from torch.nn import functional as F

from pai_act.nn.attention import MultiHeadAttention, scaled_dot_product_attention

B, H, L, S, DH = 2, 4, 3, 5, 8


def random_qkv():
    """形状 (B, H, L, DH) のクエリと (B, H, S, DH) のキー・バリューを、シード固定で作る。

    L != S にして、クエリ数とキー数の取り違え（転置の向きの誤りなど）が形状で見つかるようにする。
    """
    torch.manual_seed(0)
    return torch.randn(B, H, L, DH), torch.randn(B, H, S, DH), torch.randn(B, H, S, DH)


def padding_mask():
    """バッチ0はキーの末尾2つ、バッチ1は末尾1つを無視する。"""
    mask = torch.zeros(B, S, dtype=torch.bool)
    mask[0, -2:] = True
    mask[1, -1:] = True
    return mask


# ---- scaled_dot_product_attention ----


def test_sdpa_shapes_and_weights_sum_to_one():
    """出力と注意の重みの形状、および重みがキーの方向に足して 1 になることを確かめる。

    重みは各クエリについてのキー上の確率分布なので、softmax をかける次元（最後の次元 S）を
    取り違えると和が 1 にならない。
    """
    q, k, v = random_qkv()
    out, weights = scaled_dot_product_attention(q, k, v)
    assert out.shape == (B, H, L, DH)
    assert weights.shape == (B, H, L, S)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(B, H, L))


def test_sdpa_matches_torch():
    """マスクなしの出力が PyTorch 組み込みの F.scaled_dot_product_attention と一致する。

    √d_k でのスケーリングを忘れる・割る値を間違える、といった数式の誤りがここで見つかる。
    """
    q, k, v = random_qkv()
    out, _ = scaled_dot_product_attention(q, k, v)
    assert torch.allclose(out, F.scaled_dot_product_attention(q, k, v), atol=1e-5)


def test_sdpa_padding_mask():
    """key_padding_mask で True にしたキーへの重みが 0 になり、出力が組み込み実装と一致する。

    ACT では、CVAE エンコーダが行動チャンクの埋め草（エピソードの終わりより後ろ）を
    無視するのに使う。
    マスクの向き（True = 無視する）と、softmax の前に -inf を入れることの両方を確かめている。
    """
    q, k, v = random_qkv()
    mask = padding_mask()
    out, weights = scaled_dot_product_attention(q, k, v, key_padding_mask=mask)
    # 無視したキーへの重みは 0
    assert (weights[0, :, :, -2:] == 0).all()
    assert (weights[1, :, :, -1:] == 0).all()
    # PyTorch の attn_mask は「True = 注目してよい」で向きが逆なので反転して渡す
    expected = F.scaled_dot_product_attention(q, k, v, attn_mask=~mask[:, None, None, :])
    assert torch.allclose(out, expected, atol=1e-5)


def test_sdpa_dropout_only_in_training():
    """注意の重みへの dropout が training=True のときだけかかることを確かめる。

    評価時にも dropout がかかると、推論の出力が毎回ぶれて成功率が下がる。
    """
    q, k, v = random_qkv()
    out_eval, _ = scaled_dot_product_attention(q, k, v, dropout_p=0.5, training=False)
    out_ref, _ = scaled_dot_product_attention(q, k, v)
    assert torch.equal(out_eval, out_ref)
    out_train, _ = scaled_dot_product_attention(q, k, v, dropout_p=0.5, training=True)
    assert not torch.allclose(out_train, out_ref)


# ---- MultiHeadAttention ----

D_MODEL = 32


def make_pair(num_heads=4):
    """自前の MHA と、同じ重みを持たせた nn.MultiheadAttention の組を作る。"""
    torch.manual_seed(0)
    ours = MultiHeadAttention(D_MODEL, num_heads).eval()
    ref = nn.MultiheadAttention(D_MODEL, num_heads, batch_first=True).eval()
    # nn.MultiheadAttention は q, k, v の射影の重みを縦に連結した1つの行列 in_proj_weight で持っている
    wq, wk, wv = ref.in_proj_weight.chunk(3)
    bq, bk, bv = ref.in_proj_bias.chunk(3)
    with torch.no_grad():
        for proj, w, b in [(ours.q_proj, wq, bq), (ours.k_proj, wk, bk), (ours.v_proj, wv, bv)]:
            proj.weight.copy_(w)
            proj.bias.copy_(b)
        ours.out_proj.weight.copy_(ref.out_proj.weight)
        ours.out_proj.bias.copy_(ref.out_proj.bias)
    return ours, ref


def test_mha_self_attention_matches_torch():
    """self-attention（q = k = v）の出力が、同じ重みの nn.MultiheadAttention と一致する。

    重みが同じなのに値が合わなければ、ヘッドへの分割（view / transpose）や結合の順番に誤りがある。
    """
    ours, ref = make_pair()
    x = torch.randn(B, L, D_MODEL)
    expected, _ = ref(x, x, x)
    assert ours(x, x, x).shape == (B, L, D_MODEL)
    assert torch.allclose(ours(x, x, x), expected, atol=1e-5)


def test_mha_cross_attention_with_mask_matches_torch():
    """クエリとキーの数が違う cross-attention に padding mask を付けても、組み込み実装と一致する。

    デコーダが行動クエリ（L 個）からエンコーダの出力（S 個）を読む場面に当たる。L と S を
    取り違えた reshape や、マスクをヘッドの次元へ広げ損ねた誤りがここで見つかる。
    """
    ours, ref = make_pair()
    query = torch.randn(B, L, D_MODEL)
    memory = torch.randn(B, S, D_MODEL)
    mask = padding_mask()
    expected, _ = ref(query, memory, memory, key_padding_mask=mask)
    out = ours(query, memory, memory, key_padding_mask=mask)
    assert out.shape == (B, L, D_MODEL)
    assert torch.allclose(out, expected, atol=1e-5)


def test_mha_single_head_matches_torch():
    """ヘッドが 1 つのときも組み込み実装と一致することを確かめる。

    num_heads=1 の境界で、ヘッドの次元を潰してしまう（squeeze など）実装の誤りを見つける。
    """
    ours, ref = make_pair(num_heads=1)
    x = torch.randn(B, L, D_MODEL)
    assert torch.allclose(ours(x, x, x), ref(x, x, x)[0], atol=1e-5)
