"""Diffusion / Flow Matching で共有するデノイザ（自作の Transformer を再利用する）。

Diffusion と Flow Matching は、どちらも「ノイズの乗った行動チャンク x_t と時刻 t から、
(B, k, 14) のベクトルを出すネットワーク」を学習する。出すものの意味だけが違う:
    Diffusion:     足されたノイズ ε
    Flow Matching: ノイズから行動へ向かう速度
そこでネットワークはこの基底クラスに1つだけ書き、サブクラスは compute_loss（何を当てるか）と
predict_chunk（どう辿るか）だけを実装する。

時刻 t は「拡散過程の時刻（ノイズの量）」で、ロボットが動く時間とは関係ない:
    このファイルで「時刻 t」と書いたら、x_t にどれだけノイズが乗っているかを表す番号のこと
    （Diffusion: 整数 0〜T-1。大きいほどノイズが多い / Flow Matching: 0〜1 の実数）。
    「何番目」と呼べるものはほかに2つあり、どちらも t とは別物:
        チャンク内の位置（k 個の行動のうち何番目か。0〜k-1）  → query_pos で区別する
        エピソードの制御ステップ（シミュレータの実時間）       → モデルには入れない
    t はサンプルごとに1つで、そのサンプルの k 個の行動トークン全てに同じ値を足す
    （「チャンク全体がどれだけノイズに埋もれているか」だけを伝え、何番目の行動かは区別しない）。
    推論では、シミュレータが1ステップも進まないうちに、predict_chunk 1回の中で t を T-1 から 0 まで進める。

全体像:

    【観測の符号化】 _encode_obs: predict_chunk 1回につき1回だけ計算する
        画像 ─ backbone ─ trunk ─┐
        関節角 ─────────── trunk ─┴─ [関節角, 画像 h*w 個] ─ Encoder ─ memory

    【デノイザ】 _denoise: サンプリングのループで何度も呼ぶ（Decoder 側だけを回す）
        x_t (B, k, 14) ─ action_in_proj ─ (B, k, d) ─┐
        t (B,) ─ sinusoidal ─ time_mlp ─ (B, 1, d) ───┴─ 足す ─ Decoder（memory を cross-attention で参照）
            ─ action_out_proj ─ (B, k, 14)

ACT（policies/act.py の _decode）との違い:
    - memory に z トークンがない（動き方の違いは、推論の出発点のノイズ x_T から生まれる）
    - Decoder の入力が全て 0 ではなく、ノイズの乗った行動 x_t と時刻 t の埋め込みになる
    - それでも「何ステップ目の行動か」は x_t からは分からないので、クエリの位置ベクトル query_pos は残す

時刻 t を Encoder 側（memory）ではなく Decoder の入力に足す理由:
    サンプリングでは同じ観測のまま t だけを変えてデノイザを何度も呼ぶ。t が memory に入っていると、
    t が変わるたびに Encoder（と backbone）を計算し直すことになる。Decoder 側に入れておけば、
    観測の符号化は1回で済み、ループでは軽い Decoder だけを回せばよい。

時刻の埋め込み（DDPM 原著・Diffusion Policy と同じ形）:
    t を sinusoidal_encoding で d 次元のベクトルにし、2層の MLP（Linear → SiLU → Linear）を通す。
    sinusoidal を挟むのは、t の小さな違い（t = 3 と t = 4 など）も区別しやすいベクトルにするため。
    sinusoidal_encoding は実数の位置を受け取れるので、整数の t（Diffusion）にも、
    実数の t（Flow Matching。0〜1 を適当に拡大して渡す）にも使える。

ハイパーパラメータの既定値は ACT と同じ（比較の条件を揃えるため）:
    d_model 512 / head 8 / FFN 3200 / Encoder 4層 / Decoder 1層 / dropout 0.1
"""

import torch
from torch import nn

from pai_act.nn.backbone import ResNet18Backbone
from pai_act.nn.position import sinusoidal_encoding
from pai_act.nn.transformer import Decoder, Encoder
from pai_act.policies.base import ACTION_DIM, Policy
from pai_act.policies.trunk import ObservationTrunk


class DenoiserPolicy(Policy):
    """Diffusion / Flow Matching の基底クラス。compute_loss / predict_chunk はサブクラスが実装する。

    Args:
        chunk_size: 行動チャンクの長さ k。
        d_model: トークンの次元数。
        num_heads: attention の head 数。
        dim_feedforward: FFN の中間層の幅。
        num_encoder_layers: 観測側の Encoder の層数。
        num_decoder_layers: デノイザの Decoder の層数。
        dropout: dropout の確率。
        pretrained_backbone: ResNet18 に ImageNet 事前学習重みを使うか。
    """

    def __init__(
        self,
        chunk_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        dim_feedforward: int = 3200,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 1,
        dropout: float = 0.1,
        pretrained_backbone: bool = True,
    ):
        super().__init__(chunk_size)
        self.d_model = d_model

        # ---- 観測側（全手法で共通の部分）----
        self.backbone = ResNet18Backbone(pretrained=pretrained_backbone)
        self.trunk = ObservationTrunk(d_model)

        # Step 1: self.encoder: Encoder（num_encoder_layers 層）、self.decoder: Decoder（num_decoder_layers 層）
        #         引数の並びは ACT と同じ
        self.encoder = Encoder(num_encoder_layers, d_model, num_heads, dim_feedforward, dropout)
        self.decoder = Decoder(num_decoder_layers, d_model, num_heads, dim_feedforward, dropout)

        # Step 2: ノイズの乗った行動 → d_model の射影 self.action_in_proj: nn.Linear(ACTION_DIM, d_model)
        self.action_in_proj = nn.Linear(ACTION_DIM, d_model)

        # Step 3: 時刻の埋め込みの MLP self.time_mlp: nn.Sequential(Linear(d, d), SiLU, Linear(d, d))
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Step 4: 行動クエリの位置ベクトル self.query_pos（学習する）: nn.Parameter で (1, chunk_size, d_model)、
        #         初期値は torch.randn（ACT の既定と同じ）
        self.query_pos = nn.Parameter(torch.randn(1, chunk_size, d_model))

        # Step 5: Decoder の出力 → (B, k, 14) の射影 self.action_out_proj: nn.Linear(d_model, ACTION_DIM)
        self.action_out_proj = nn.Linear(d_model, ACTION_DIM)

    def _embed_time(self, t: torch.Tensor) -> torch.Tensor:
        """時刻 t をトークンに足せる形のベクトルにする。

        Args:
            t: (B,) 各サンプルの拡散過程の時刻（行動の何番目かではない）。
                整数（Diffusion）でも実数（Flow Matching）でもよい。

        Returns:
            (B, 1, d_model)。k 個の行動トークンに同じ値を足せるよう、真ん中に長さ 1 の軸を入れる。
        """
        # Step 1: sinusoidal_encoding(t を float にしたもの, self.d_model) で (B, d) にする
        time_emb = sinusoidal_encoding(t.float(), self.d_model)  # (B, d_model)

        # Step 2: self.time_mlp を通し、unsqueeze で (B, 1, d) にする
        time_emb = self.time_mlp(time_emb).unsqueeze(1)  # (B, 1, d_model)

        return time_emb

    def _encode_obs(
        self, image: torch.Tensor, state_n: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """観測を memory にする（ACT の _decode の Step 1・3 から z を除いたもの）。

        Args:
            image: (B, 3, H, W) uint8
            state_n: (B, 14) 正規化済みの関節角。

        Returns:
            memory: (B, 1+h*w, d)  [関節角, 画像 h*w 個] を Encoder に通したもの
            memory_pos: (1, 1+h*w, d)  memory の位置（Decoder の cross-attention で使う）
        """
        # Step 1: 画像を self.backbone に通し、self.trunk で観測トークン列と位置を得る
        feature = self.backbone(image)  # (B, in_channels, h, w)
        obs_tokens, obs_pos = self.trunk(feature, state_n)  # (B, 1+h*w, d), (1, 1+h*w, d)

        # Step 2: self.encoder に通して memory にする。位置は trunk の位置をそのまま memory_pos として返す
        memory = self.encoder(obs_tokens, pos=obs_pos)  # (B, 1+h*w, d)
        memory_pos = obs_pos  # (1, 1+h*w, d)

        return memory, memory_pos

    def _denoise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        memory: torch.Tensor,
        memory_pos: torch.Tensor,
    ) -> torch.Tensor:
        """ノイズの乗った行動チャンクと時刻から、(B, k, 14) を出す（Diffusion では ε の予測）。

        Args:
            x_t: (B, k, 14) ノイズの乗った行動チャンク（正規化済みの空間）。
            t: (B,) 拡散過程の時刻（ノイズの量）。k 個の行動トークンに共通。
            memory, memory_pos: _encode_obs の出力。

        Returns:
            pred: (B, k, 14) x_t・t・観測（memory）から予測したベクトル
                （Diffusion では ε̂、Flow Matching では速度）。
        """
        # Step 1: x_t を self.action_in_proj で (B, k, d) にする
        x_t_tokens = self.action_in_proj(x_t)  # (B, k, d)

        # Step 2: self._embed_time(t) の (B, 1, d) を足す（ブロードキャストで k 個全てに同じ値が足される）
        time_emb = self._embed_time(t)  # (B, 1, d)
        x_t_tokens = x_t_tokens + time_emb  # (B, k, d)

        # Step 3: self.decoder に通す（pos = self.query_pos、memory_pos = memory_pos）
        decoder_out = self.decoder(
            x_t_tokens, memory, pos=self.query_pos, memory_pos=memory_pos
        )  # (B, k, d)

        # Step 4: self.action_out_proj で (B, k, 14) にする
        pred = self.action_out_proj(decoder_out)  # (B, k, 14)

        return pred
