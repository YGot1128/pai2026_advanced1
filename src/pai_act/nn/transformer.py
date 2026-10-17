"""Transformer の Encoder / Decoder（自前実装。nn.Transformer 系は使わない）。

ACT は DETR と同じ構成の Transformer を使う:

    EncoderLayer（観測トークン同士で情報を混ぜる）:
        x ─ self-attention ─ 残差 + LayerNorm ─ FFN ─ 残差 + LayerNorm ─ x

    DecoderLayer（k 個の行動クエリが、Encoder の出力 memory から情報を取ってくる）:
        x ─ self-attention ─ 残差 + LayerNorm
          ─ cross-attention（K, V は memory）─ 残差 + LayerNorm
          ─ FFN ─ 残差 + LayerNorm ─ x

部品の役割:
    残差接続 x + f(x): 層を重ねても勾配が入力側まで届くようにする（f が何も学ばなくても恒等写像が残る）
    LayerNorm: トークンごとに特徴ベクトルを標準化し、層を通るたびに値のスケールが崩れるのを防ぐ
    FFN: トークンごとに独立にかける2層 MLP（d_model → dim_feedforward → d_model）。
         attention がトークン「間」で情報を混ぜ、FFN が各トークン「内」で特徴を変換する、という分担

位置エンコーディングの入れ方（DETR 方式）:
    原著 Transformer は入力に位置エンコーディングを1回足すだけだが、DETR / ACT は
    **各層の attention で、クエリとキーにだけ** 位置を足す（バリューには足さない）:
        q = k = x + pos,   v = x
    位置は「どこに注目するか」を決めるのに使い、運ぶ中身（バリュー）には混ぜない、という考え方。
    層ごとに足し直すので、深い層でも位置の情報が薄れない。

post-norm と pre-norm:
    上の図は「残差を足した後に LayerNorm」をかける post-norm（原著 Transformer・ACT 公式実装の既定）。
    「サブ層に入れる前に LayerNorm」をかける pre-norm は深いモデルで学習が安定しやすいが、
    ここでは ACT に合わせて post-norm にする。

dropout は「attention の重み」と「残差に足す直前のサブ層の出力」と「FFN の中間」にかける。
"""

import torch
from torch import nn

from pai_act.nn.attention import MultiHeadAttention


def _with_pos(x: torch.Tensor, pos: torch.Tensor | None) -> torch.Tensor:
    """位置エンコーディングがあれば足す（クエリ・キー用）。"""
    return x if pos is None else x + pos


class FeedForward(nn.Module):
    """トークンごとの2層 MLP: Linear(d_model → dim_ff) → ReLU → Dropout → Linear(dim_ff → d_model)。"""

    def __init__(self, d_model: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        # Step 1: 上の docstring の4つの層を作る
        self.net = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, d_model) → (B, N, d_model)。nn.Linear は最後の軸にだけ作用するので、形はそのままでよい。"""
        return self.net(x)


class EncoderLayer(nn.Module):
    """Encoder の1層（self-attention + FFN、post-norm）。

    Args:
        d_model: トークンの次元数。
        num_heads: attention の head 数。
        dim_feedforward: FFN の中間層の幅。
        dropout: dropout の確率。
    """

    def __init__(self, d_model: int, num_heads: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        # Step 1: self-attention: MultiHeadAttention（dropout は注意の重みにかかる）
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)

        # Step 2: FFN: FeedForward
        self.ffn = FeedForward(d_model, dim_feedforward, dropout)

        # Step 3: LayerNorm を2つ（attention の後用 norm1、FFN の後用 norm2）
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Step 4: 残差に足す直前にかける nn.Dropout（attention 用・FFN 用の2つ）
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, d_model) トークン列。
            pos: (B, N, d_model) または (1, N, d_model) の位置エンコーディング。None なら足さない。
            key_padding_mask: (B, N) bool。True のトークンはキーとして無視される。

        Returns:
            (B, N, d_model)
        """
        # Step 1: クエリとキーに位置を足したもの q = k = _with_pos(x, pos) を作る。バリューは x のまま
        q = k = _with_pos(x, pos)
        v = x

        # Step 2: self-attention → dropout → 残差 (x + ...) → norm1
        self_attn_out = self.self_attention(q, k, v, key_padding_mask=key_padding_mask)
        self_attn_out = self.dropout1(self_attn_out)
        x = x + self_attn_out
        x = self.norm1(x)

        # Step 3: FFN → dropout → 残差 → norm2
        ffn_out = self.ffn(x)
        ffn_out = self.dropout2(ffn_out)
        x = x + ffn_out
        x = self.norm2(x)

        return x


class DecoderLayer(nn.Module):
    """Decoder の1層（self-attention + cross-attention + FFN、post-norm）。引数は EncoderLayer と同じ。"""

    def __init__(self, d_model: int, num_heads: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        # Step 1: self-attention と cross-attention（MultiHeadAttention を2つ）
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)

        # Step 2: FFN: FeedForward
        self.ffn = FeedForward(d_model, dim_feedforward, dropout)

        # Step 3: LayerNorm を3つ（self-attention 後 / cross-attention 後 / FFN 後）
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Step 4: 残差に足す直前にかける nn.Dropout を3つ
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        pos: torch.Tensor | None = None,
        memory_pos: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model) Decoder 側のトークン列（ACT では k 個の行動クエリ）。
            memory: (B, S, d_model) Encoder の出力。
            pos: x の位置エンコーディング（(B or 1, L, d_model) か None）。
            memory_pos: memory の位置エンコーディング（(B or 1, S, d_model) か None）。
            memory_key_padding_mask: (B, S) bool。True の memory トークンは無視される。

        Returns:
            (B, L, d_model)
        """
        # Step 1: self-attention（q = k = x + pos, v = x）→ dropout → 残差 → LayerNorm
        q = k = _with_pos(x, pos)
        v = x
        self_attn_out = self.self_attention(q, k, v)
        self_attn_out = self.dropout1(self_attn_out)
        x = x + self_attn_out
        x = self.norm1(x)

        # Step 2: cross-attention → dropout → 残差 → LayerNorm
        #         クエリは x + pos、キーは memory + memory_pos、バリューは memory
        q = _with_pos(x, pos)
        k = _with_pos(memory, memory_pos)
        v = memory
        cross_attn_out = self.cross_attention(q, k, v, key_padding_mask=memory_key_padding_mask)
        cross_attn_out = self.dropout2(cross_attn_out)
        x = x + cross_attn_out
        x = self.norm2(x)

        # Step 3: FFN → dropout → 残差 → LayerNorm
        ffn_out = self.ffn(x)
        ffn_out = self.dropout3(ffn_out)
        x = x + ffn_out
        x = self.norm3(x)

        return x


class Encoder(nn.Module):
    """EncoderLayer を num_layers 個重ねたもの。EncoderLayer.forward と同じ引数で呼ぶ。"""

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Step 1: EncoderLayer を num_layers 個作り、nn.ModuleList に入れる
        #         （Python の list だと中の層のパラメータが model.parameters() に含まれず、学習されない）
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, dim_feedforward, dropout) for _ in range(num_layers)]
        )

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Step 1: 各層に順に通す（pos と key_padding_mask は全層に同じものを渡す）
        for layer in self.layers:
            x = layer(x, pos, key_padding_mask)

        return x


class Decoder(nn.Module):
    """DecoderLayer を num_layers 個重ね、最後に LayerNorm をかけたもの。

    最後の LayerNorm は DETR / ACT 公式実装に合わせたもの（post-norm なので各層の出力は既に
    LayerNorm 済みだが、公式実装と同じ構成にしておく）。引数は __init__ が Encoder と、
    forward が DecoderLayer.forward と同じ。
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Step 1: DecoderLayer を num_layers 個作り、nn.ModuleList に入れる
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, dim_feedforward, dropout) for _ in range(num_layers)]
        )
        # Step 2: 最後にかける LayerNorm を作る
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        pos: torch.Tensor | None = None,
        memory_pos: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Step 1: 各層に順に通す（memory・位置・マスクは全層に同じものを渡す）
        for layer in self.layers:
            x = layer(x, memory, pos, memory_pos, memory_key_padding_mask)

        # Step 2: 最後の LayerNorm をかけて返す
        x = self.norm(x)

        return x
