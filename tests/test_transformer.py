"""nn/transformer.py のテスト。

uv run pytest tests/test_transformer.py -v
"""

import torch

from pai_act.nn.transformer import Decoder, DecoderLayer, Encoder, EncoderLayer, FeedForward

B, N, M, D, HEADS, FF = 2, 6, 4, 32, 4, 64


def inputs():
    """トークン列 (B, N, D) と、バッチで共有する位置エンコーディング (1, N, D) を作る。"""
    torch.manual_seed(0)
    return torch.randn(B, N, D), torch.randn(1, N, D)


def test_feedforward_shape():
    """FeedForward は形状を変えず、Linear(D→FF) と Linear(FF→D) の2層だけを持つ。

    パラメータ数を数えることで、余分な層や、バイアスの付け忘れを見つける。
    """
    ff = FeedForward(D, FF, dropout=0.0)
    assert ff(torch.randn(B, N, D)).shape == (B, N, D)
    assert sum(p.numel() for p in ff.parameters()) == (D * FF + FF) + (FF * D + D)


def test_encoder_layer_shape_and_post_norm():
    """EncoderLayer は形状を変えず、post-norm（残差を足した後に LayerNorm）になっている。

    ACT・DETR の既定は post-norm。pre-norm だと出力の最後が残差の足し算になり、
    各トークンが平均0・分散1にならないので区別できる。
    """
    layer = EncoderLayer(D, HEADS, FF, dropout=0.0).eval()
    x, pos = inputs()
    y = layer(x, pos)
    assert y.shape == (B, N, D)
    # post-norm: 最後に LayerNorm（初期値は重み1・バイアス0）を通るので、各トークンは平均0・分散1
    assert torch.allclose(y.mean(-1), torch.zeros(B, N), atol=1e-5)
    assert torch.allclose(y.var(-1, unbiased=False), torch.ones(B, N), atol=1e-3)


def test_encoder_without_pos_is_permutation_equivariant():
    """位置エンコーディングなしの Encoder は、入力を並べ替えると出力も同じように並べ替わるだけ。

    Attention はトークンの順番を区別できない（位置エンコーディングが必要な理由）。これが
    成り立たなければ、トークンをまたいで混ぜる処理（次元の取り違えなど）が紛れ込んでいる。
    """
    enc = Encoder(2, D, HEADS, FF, dropout=0.0).eval()
    x, _ = inputs()
    perm = torch.randperm(N)
    assert torch.allclose(enc(x)[:, perm], enc(x[:, perm]), atol=1e-5)


def test_encoder_uses_pos():
    """Encoder に渡した位置エンコーディングが出力に反映される（引数の渡し忘れを見つける）。"""
    enc = Encoder(2, D, HEADS, FF, dropout=0.0).eval()
    x, pos = inputs()
    assert not torch.allclose(enc(x, pos), enc(x))


def test_encoder_key_padding_mask():
    """マスクで無視したトークンの中身を変えても、他のトークンの出力が変わらないことを確かめる。

    key_padding_mask が全ての層の self-attention まで渡っているかを確かめている。
    """
    enc = Encoder(2, D, HEADS, FF, dropout=0.0).eval()
    x, pos = inputs()
    mask = torch.zeros(B, N, dtype=torch.bool)
    mask[:, -2:] = True
    x2 = x.clone()
    x2[:, -2:] = torch.randn(B, 2, D)
    assert torch.allclose(enc(x, pos, mask)[:, :-2], enc(x2, pos, mask)[:, :-2], atol=1e-5)


def test_decoder_layer_shape():
    """DecoderLayer の出力はクエリ側の形状 (B, M, D) になる（memory 側の長さ N ではない）。"""
    layer = DecoderLayer(D, HEADS, FF, dropout=0.0).eval()
    x, memory = torch.randn(B, M, D), torch.randn(B, N, D)
    assert layer(x, memory, torch.randn(1, M, D), torch.randn(1, N, D)).shape == (B, M, D)


def test_decoder_reads_memory():
    """Decoder が cross-attention で memory を読み、クエリの位置ベクトルでクエリを区別する。

    ACT のデコーダは入力 x を全て 0 にし、行動クエリの違いを位置ベクトル（query_pos）だけで
    表す。そのため query_pos が各層で足されていないと、チャンク内の全ステップが同じ行動になる。
    """
    dec = Decoder(1, D, HEADS, FF, dropout=0.0).eval()
    x, query_pos = torch.zeros(B, M, D), torch.randn(1, M, D)
    memory, memory_pos = inputs()
    y = dec(x, memory, query_pos, memory_pos)
    assert y.shape == (B, M, D)
    # cross-attention で memory を参照しているので、memory が変われば出力も変わる
    assert not torch.allclose(y, dec(x, memory + 1.0, query_pos, memory_pos))
    # 入力 x が全て0でも、クエリの位置ベクトルが違えば出力が違う（ACT の行動クエリはこの仕組み）
    assert not torch.allclose(y[:, 0], y[:, 1])


def test_layers_are_registered_and_receive_grad():
    """層が nn.ModuleList に登録され、エンコーダ・デコーダの全パラメータに勾配が流れる。

    Python の list に入れた層はパラメータとして登録されず、optimizer に渡らない（学習されない）。
    勾配が届かないパラメータがあれば、forward で使われていない部品がある。
    """
    enc = Encoder(3, D, HEADS, FF)
    dec = Decoder(2, D, HEADS, FF)
    # nn.ModuleList に入れていれば、層の数に比例してパラメータが増える
    assert sum(p.numel() for p in enc.parameters()) == 3 * sum(
        p.numel() for p in EncoderLayer(D, HEADS, FF).parameters()
    )
    x, pos = inputs()
    memory = enc(x, pos)
    dec(torch.zeros(B, M, D), memory, torch.randn(1, M, D), pos).sum().backward()
    for name, p in [*enc.named_parameters(), *dec.named_parameters()]:
        assert p.grad is not None, f"{name} に勾配が流れていない"
