"""Multi-Head Attention（自前実装。nn.MultiheadAttention / F.scaled_dot_product_attention は使わない）。

Attention は「クエリ（問い合わせ）ごとに、キーとの類似度で重みを決め、バリューの重み付き平均を取る」操作:

    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

    Q: (L, d_k)  L 個のクエリ   … 「何を探しているか」
    K: (S, d_k)  S 個のキー     … 「自分は何か」（クエリとの内積で関連度を測る）
    V: (S, d_v)  S 個のバリュー … 「関連すると判断されたときに渡す中身」
    出力: (L, d_v) … クエリごとに、S 個のバリューの重み付き平均

sqrt(d_k) で割る理由:
    q, k の各成分が平均 0・分散 1 なら、内積 q·k の分散は d_k になる。d_k が大きいと内積の値が大きくなり、
    softmax がほぼ one-hot に飽和して勾配が消える。sqrt(d_k) で割ると分散が 1 程度に保たれる。

Multi-Head:
    d_model 次元を H 個の head に分け（各 head の次元 Dh = d_model / H）、head ごとに別々の Attention を
    計算してから結合する。head ごとに「どこに注目するか」を変えられる
    （例: ある head はキューブの位置、別の head はグリッパーの位置に注目する）。
    計算量は 1 head で d_model 次元の Attention をするのとほぼ同じ。

ACT での使われ方:
    - self-attention: Q, K, V がすべて同じトークン列から来る（Encoder・Decoder の各層）
    - cross-attention: Q は Decoder 側、K と V は Encoder の出力（Decoder の各層）
    - key_padding_mask: CVAE エンコーダで、エピソード外の埋め草の行動（action_is_pad）を無視するのに使う
    - ACT では因果マスク（未来のトークンを見せないマスク）は使わない。Decoder は k 個の行動を
      自己回帰ではなく一度に出すため、k 個のクエリは互いに全部を見てよい。

形状の約束（このリポジトリ全体で共通）: バッチが先頭の (B, 系列長, 次元)。

形状の記号（括弧内は ACT の既定値での例）:
    B       バッチの大きさ。サンプルごとに独立に計算する（学習時 8）
    H       head の数（8）
    L       クエリのトークン数 = 出力のトークン数。出力はクエリ 1 個につき 1 トークン
    S       キー・バリューのトークン数。k と v は必ず同じ数（キー j とバリュー j が対になる）
    d_model トークン 1 個のベクトルの次元（512）
    Dh      head あたりの次元 = d_model / H（512 / 8 = 64）

    L と S は使う場所で決まる:
        Encoder の self-attention        L = S = 302（z 1 + 関節角 1 + 画像 15x20=300）
        CVAE エンコーダの self-attention L = S = 102（CLS 1 + 関節角 1 + 行動 k=100）
        Decoder の self-attention        L = S = 100（行動チャンクの長さ k）
        Decoder の cross-attention       L = 100（Decoder 側）、S = 302（Encoder の出力）
    self-attention では L = S、cross-attention では一般に L ≠ S。
    注意の重み (B, H, L, S) は「L 個のクエリそれぞれが、S 個のキーのどれをどれだけ見るか」の表。
"""

import torch
from torch import nn
from torch.nn import functional as F


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    key_padding_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    training: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """softmax(q k^T / sqrt(Dh)) v を head ごとに計算する。

    B と H の軸はバッチとして扱われ、(b, h) の組ごとに独立に (L, Dh) と (S, Dh) の Attention を計算する。
    記号の意味はモジュールの docstring の「形状の記号」を参照。

    Args:
        q: (B, H, L, Dh)
        k: (B, H, S, Dh)
        v: (B, H, S, Dh)
        key_padding_mask: (B, S) bool。True のキーは無視する（注意の重みを 0 にする）。
            nn.MultiheadAttention の key_padding_mask と同じ向き（True = 無視）。
        dropout_p: 注意の重みにかける dropout の確率。
        training: True のときだけ dropout をかける（推論時は dropout しない）。

    Returns:
        out: (B, H, L, Dh)
        weights: (B, H, L, S) softmax 後・dropout 前の注意の重み（各クエリについて S 方向の和が 1）。
            学習には使わないが、どこに注目しているかの可視化に使える。

    Note:
        あるクエリから見て全てのキーが無視されると、softmax の分母が 0 になり NaN が出る。
        ACT では CLS トークンなど必ず有効なキーがあるので、この場合は考えなくてよい。
    """
    # Step 1: スコア = q と k の内積を sqrt(Dh) で割ったもの。形は (B, H, L, S)
    score = torch.matmul(q, k.transpose(-2, -1)) / q.size(-1) ** 0.5  # (B, H, L, S)

    # Step 2: key_padding_mask があれば、True の位置のスコアを -inf にする
    if key_padding_mask is not None:
        score = score.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))

    # Step 3: 最後の軸（S 方向）で softmax を取り、注意の重み weights にする
    # S 方向で取るのは、重みが「クエリ 1 個が S 個のキーにどう配分するか」だから。
    # score[b, h, i, :] がクエリ i の全キーへのスコアで、これを和 1 の確率分布にすると、
    # 次の weights @ v がクエリごとのバリューの重み付き平均になる。
    # L 方向で取ると「キー 1 個が各クエリにどう配分されるか」になり、クエリごとの和が 1 にならない。
    # Step 2 で -inf にしたキーは exp(-inf) = 0 なので、重みが 0 になり平均に入らない。
    weights = F.softmax(score, dim=-1)  # (B, H, L, S)

    # Step 4: weights に dropout をかけたものを作る（dropout_p と training を渡す。F.dropout が使える）
    weights_dropout = F.dropout(weights, p=dropout_p, training=training)

    # Step 5: dropout 後の重みと v の行列積で出力を作る。形は (B, H, L, Dh)
    out = torch.matmul(weights_dropout, v)  # (B, H, L, Dh)

    return out, weights


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention。

        query (B, L, d) ─ q_proj ─┐
        key   (B, S, d) ─ k_proj ─┼─ head に分割 ─ scaled_dot_product_attention ─ head を結合 ─ out_proj ─ (B, L, d)
        value (B, S, d) ─ v_proj ─┘

    Args:
        d_model: トークンの次元数。num_heads で割り切れること。
        num_heads: head の数 H。
        dropout: 注意の重みにかける dropout の確率。
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        # Step 1: d_model が num_heads で割り切れることを assert で確認し、head あたりの次元 head_dim を求める
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        head_dim = d_model // num_heads

        # Step 2: 線形層を4つ作る（どれも d_model → d_model、bias あり）
        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

        # Step 3: num_heads, head_dim, dropout を self の属性として保存する（forward で使う）
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """query (B, L, d_model), key / value (B, S, d_model) → (B, L, d_model)。

        key_padding_mask: (B, S) bool。True のキーは無視する。
        """
        B, L, _ = query.size()
        # Step 1: q_proj / k_proj / v_proj で射影する（形は変わらない）
        query_proj = self.q_proj(query)  # (B, L, d_model)
        key_proj = self.k_proj(key)  # (B, S, d_model)
        value_proj = self.v_proj(value)  # (B, S, d_model)

        # Step 2: 各テンソルを head に分ける: (B, L, d_model) → (B, L, H, Dh) → (B, H, L, Dh)
        def _split_heads(s: torch.Tensor) -> torch.Tensor:
            _B, _L, _ = s.size()
            return s.view(_B, _L, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L, Dh)

        query_heads = _split_heads(query_proj)  # (B, H, L, Dh)
        key_heads = _split_heads(key_proj)  # (B, H, S, Dh)
        value_heads = _split_heads(value_proj)  # (B, H, S, Dh)

        # Step 3: scaled_dot_product_attention を呼ぶ
        out, _ = scaled_dot_product_attention(
            query_heads, key_heads, value_heads, key_padding_mask, self.dropout, self.training
        )  # (B, H, L, Dh)
        # Step 4: head を結合して元の形に戻す: (B, H, L, Dh) → (B, L, H, Dh) → (B, L, d_model)
        # 注意: transpose したテンソルはメモリ上で連続している必要がある
        # view の前に.contiguous() が要る（reshape なら自動でやってくれる）
        out = out.transpose(1, 2).reshape(B, L, -1)  # (B, L, d_model)

        # Step 5: out_proj をかけて返す
        return self.out_proj(out)  # (B, L, d_model)
