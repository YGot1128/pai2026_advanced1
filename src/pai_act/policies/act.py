"""ACT（Action Chunking with Transformers, Zhao et al. 2023）: CVAE ヘッド。

全体像（学習時）:

    【CVAE エンコーダ】デモの行動列から「動き方のスタイル」を表す潜在変数 z を推定する（学習時だけ使う）
        [CLS, 関節角, 行動 a_t..a_{t+k-1}] (k+2 個) ─ Encoder ─ CLS の位置の出力 ─ Linear ─ μ, log σ²
        z = μ + σ ε,  ε ~ N(0, I)                                      … reparameterization trick

    【デコーダ（= 方策本体）】観測と z から行動チャンクを出す
        [z, 関節角, 画像 h*w 個] ─ Encoder ─ memory
        k 個の行動クエリ ─ Decoder（memory を cross-attention で参照）─ Linear ─ 行動チャンク (B, k, 14)

    損失 = L1(予測チャンク, 正解チャンク) + β · KL( N(μ, σ²) || N(0, I) )

推論時: CVAE エンコーダは使わず（これから実行する行動は分からないので）、z = 0（事前分布 N(0, I) の平均）とする。

なぜ CVAE か:
    人間のデモは同じ状況でも動きが揺らぐ（速く掴む・ゆっくり掴む など）。決定的な回帰ではその平均を
    学んでしまい、どのデモにも似ていない中途半端な動きになる。z に「このデモはどういう動き方か」を
    持たせると、デコーダは z を手がかりに一貫した動きを出せる。KL 項は z の分布を N(0, I) に近づけ、
    推論時に z = 0 を入れても自然な動きになるようにする。
    CVAE エンコーダの入力に画像を入れないのは、計算を軽くするためと、z に観測ではなく
    「動き方」の情報だけを持たせるため。

ハイパーパラメータの既定値は ACT 原著の sim 実験と LeRobot の ACT 設定に合わせる:
    d_model 512 / head 8 / FFN 3200 / CVAE エンコーダ 4層 / Encoder 4層 / Decoder 1層（※） /
    z 32次元 / β (kl_weight) 10 / dropout 0.1

※ Decoder の層数: 原著・公式実装の設定は 7 層だが、公式実装のバグで最初の1層の出力しか使われていない
   （https://github.com/tonyzhaozh/act/issues/25 。LeRobot はこれに合わせて 1 層にしている）。
   ここでも既定値は 1 とする。層数を変えた比較はレポートの考察材料にできる。

正規化: ポリシーの外（学習ループ・評価ループ）とやり取りする関節角・行動は、物理シミュレータ
（gym-aloha）の単位（rad など、正規化前の値）のままにする（policies/base.py の約束）。
ポリシーの内部では、関節角・行動を平均 0・標準偏差 1 に正規化してからニューラルネットワークに
入出力させ、損失も正規化した行動の上で計算する（baseline と同じ）。
"""

import torch
from torch import nn

from pai_act.nn.backbone import ResNet18Backbone
from pai_act.nn.position import sinusoidal_1d
from pai_act.nn.transformer import Decoder, Encoder
from pai_act.policies.base import ACTION_DIM, STATE_DIM, Policy, masked_l1
from pai_act.policies.trunk import ObservationTrunk


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """N(μ, σ²) から z をサンプルする（reparameterization trick）。

    z を N(μ, σ²) から直接サンプルすると、「サンプルする」という操作を通して μ, σ に勾配が流れない。
    乱数 ε ~ N(0, I) を外から与え、z = μ + σ ε と μ, σ の微分可能な式で書くと、μ と σ に勾配が流れる。

    ニューラルネットワークが σ ではなく log σ² を出すのは、σ > 0 の制約を気にせず任意の実数を出力でき、数値的にも安定するため。

    Args:
        mu, logvar: (B, latent_dim) 平均 μ と log σ²。

    Returns:
        z: (B, latent_dim)
    """
    # Step 1: σ = exp(log σ² / 2) を計算する
    sigma = torch.exp(logvar / 2)  # (B, latent_dim)

    # Step 2: ε を mu と同じ形で、標準正規分布の乱数として生成する（torch.randn_like）
    epsilon = torch.randn_like(mu)  # (B, latent_dim)

    # Step 3: z = μ + σ ε を返す
    z = mu + sigma * epsilon  # (B, latent_dim)

    return z


def kl_to_standard_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL( N(μ, σ²) || N(0, I) ) を潜在次元について和を取り、バッチで平均したスカラー。

    1次元あたりの閉形式（ガウス分布どうしの KL は積分せずに式で書ける）:
        KL = -1/2 · (1 + log σ² - μ² - σ²)
    μ = 0, σ = 1 のとき 0 になり、そこから離れるほど大きくなる。

    Args:
        mu, logvar: (B, latent_dim)

    Returns:
        スカラー
    """
    # Step 1: 上の式を要素ごとに計算する → (B, latent_dim)。σ² は exp(log σ²)
    kl = -0.5 * (1 + logvar - mu**2 - torch.exp(logvar))  # (B, latent_dim)
    # Step 2: 潜在次元（最後の軸）で和を取る → (B,)
    kl_sum = kl.sum(dim=-1)  # (B,)

    # Step 3: バッチで平均を取る → スカラー
    kl_mean = kl_sum.mean()  # スカラー

    return kl_mean


class ACTPolicy(Policy):
    """ACT。引数の意味はモジュール docstring の「ハイパーパラメータ」を参照。

    Args:
        chunk_size: 行動チャンクの長さ k。
        d_model: トークンの次元数。
        num_heads: attention の head 数。
        dim_feedforward: FFN の中間層の幅。
        num_encoder_layers: 方策本体の Encoder の層数。
        num_decoder_layers: 方策本体の Decoder の層数。
        num_vae_encoder_layers: CVAE エンコーダの層数。
        latent_dim: 潜在変数 z の次元数。
        dropout: dropout の確率。
        kl_weight: 損失の KL 項の重み β。
        learned_query_pos: 行動クエリの位置ベクトルを学習させるか。True が ACT 公式実装・LeRobot、
            False が論文（4章・Fig. 11）の記述どおりの固定 sinusoidal（追加実験用）。
        pretrained_backbone: ResNet18 に ImageNet 事前学習重みを使うか。
    """

    # パラメータにもバッファにもなる属性は、型チェッカーに Tensor だと伝えるために宣言しておく
    query_pos: torch.Tensor

    def __init__(
        self,
        chunk_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        dim_feedforward: int = 3200,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 1,
        num_vae_encoder_layers: int = 4,
        latent_dim: int = 32,
        dropout: float = 0.1,
        kl_weight: float = 10.0,
        learned_query_pos: bool = True,
        pretrained_backbone: bool = True,
    ):
        # このclassの親クラスは、base.pyのPolicyクラスであるため、
        # Policy.__init__が要求するchunk_sizeを渡す。
        super().__init__(chunk_size)
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight

        # ---- 観測側（全手法で共通の部分）----
        self.backbone = ResNet18Backbone(pretrained=pretrained_backbone)
        self.trunk = ObservationTrunk(d_model)

        # ---- CVAE エンコーダ ----
        # Step 1: CLS トークン self.cls_token（学習するベクトル）: nn.Parameter で (1, 1, d_model)、初期値は torch.randn
        #         「デモ全体の要約」を集めるための席。BERT や ViT の [CLS] と同じ考え方
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Step 2: 関節角 → d_model、行動 → d_model の射影（nn.Linear を2つ）
        #         self.vae_state_proj, self.vae_action_proj（trunk の射影とは別に持つ）
        self.vae_state_proj = nn.Linear(STATE_DIM, d_model)
        self.vae_action_proj = nn.Linear(ACTION_DIM, d_model)

        # Step 3: self.vae_encoder: Encoder（num_vae_encoder_layers 層）
        self.vae_encoder = Encoder(
            num_vae_encoder_layers, d_model, num_heads, dim_feedforward, dropout
        )

        # Step 4: CLS の出力 → (μ, log σ²) の射影 self.latent_proj: nn.Linear(d_model, 2 * latent_dim)
        self.latent_proj = nn.Linear(d_model, 2 * latent_dim)

        # Step 5: 入力列（長さ chunk_size + 2）の 1D 位置エンコーディングを sinusoidal_1d で作り、
        #         形 (1, chunk_size + 2, d_model) にして self.register_buffer("vae_pos", ...) で持つ
        #         chunk_size + 2 のうち、2は CLS と関節角の席。残り chunk_size は行動チャンクの席
        #         ※ バッファにすると policy.to(device) で一緒に移動し、学習の対象にはならない
        input_sequence = sinusoidal_1d(chunk_size + 2, d_model)  # (chunk_size + 2, d_model)
        input_sequence = input_sequence.unsqueeze(0)  # (1, chunk_size + 2, d_model)
        # 属性名 vae_pos は、CVAE エンコーダの forward で pos 引数として渡すときに使う
        # persistent=False にすると、state_dict に含まれず、学習済みモデルの保存・読み込みの対象にならない
        self.register_buffer("vae_pos", input_sequence, persistent=False)

        # ---- 方策本体 ----
        # Step 6: z → d_model の射影 self.latent_in_proj: nn.Linear(latent_dim, d_model)
        self.latent_in_proj = nn.Linear(latent_dim, d_model)

        # Step 7: z トークンの位置ベクトル self.latent_pos（学習する）: nn.Parameter で (1, 1, d_model)
        self.latent_pos = nn.Parameter(torch.randn(1, 1, d_model))  # (1, 1, d_model)

        # Step 8: self.encoder: Encoder（num_encoder_layers 層）、self.decoder: Decoder（num_decoder_layers 層）
        self.encoder = Encoder(num_encoder_layers, d_model, num_heads, dim_feedforward, dropout)
        self.decoder = Decoder(num_decoder_layers, d_model, num_heads, dim_feedforward, dropout)

        # Step 9: 行動クエリの位置ベクトル self.query_pos: 形 (1, chunk_size, d_model)
        #         k 個のクエリそれぞれが「何ステップ先の行動を担当するか」を表す。
        #         learned_query_pos で2通りに作り分ける:
        #         - True（既定）: nn.Parameter で学習する。初期値は torch.randn
        #           （ACT 公式実装・LeRobot の nn.Embedding と同じ。重みの初期化も標準正規分布）
        #         - False: sinusoidal_1d の固定値をバッファ（register_buffer）で持つ
        #           （論文 4章・Fig. 11 の記述どおり。追加実験用: ACT_IMPLEMENTATION.md の 10.）
        #         どちらも self.query_pos で読めるので、_decode 側は分岐しなくてよい

        # 位置ベクトルを学習させる場合
        if learned_query_pos:
            self.query_pos = nn.Parameter(
                torch.randn(1, chunk_size, d_model)
            )  # (1, chunk_size, d_model)
        # 位置ベクトルを固定する場合
        else:
            query_pos = sinusoidal_1d(chunk_size, d_model)
            query_pos = query_pos.unsqueeze(0)  # (1, chunk_size, d_model)
            self.register_buffer("query_pos", query_pos, persistent=False)
        # Step 10: Decoder の出力 → 行動 self.action_head: nn.Linear(d_model, ACTION_DIM)
        self.action_head = nn.Linear(d_model, ACTION_DIM)

        # Step 11（任意）: Encoder / Decoder の重み（2次元以上のパラメータ）を nn.init.xavier_uniform_ で
        #         初期化し直す（DETR / ACT 公式実装と同じ）。
        # まずは無しで学習し、成功率を見てから比べる

    def _encode_latent(
        self, state_n: torch.Tensor, action_n: torch.Tensor, action_is_pad: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """CVAE エンコーダ。

        Args:
            state_n: (B, 14) 正規化済みの関節角。
            action_n: (B, k, 14) 正規化済みの行動チャンク（正解）。
            action_is_pad: (B, k) bool。True のステップはエピソード外の埋め草。

        Returns:
            mu, logvar: 各 (B, latent_dim)
        """
        # Step 1: 入力トークン列を作る: [CLS (B,1,d), 関節角 (B,1,d), 行動 (B,k,d)] → (B, k+2, d)
        #         CLS はバッチの数だけ並べる（self.cls_token.expand(B, -1, -1)）
        batch_size = state_n.shape[0]
        input_cls_token = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, d)
        input_state_proj = self.vae_state_proj(state_n).unsqueeze(1)  # (B, 1, d)
        input_action_proj = self.vae_action_proj(action_n)  # (B, k, d)
        input_tokens = torch.cat(
            [input_cls_token, input_state_proj, input_action_proj], dim=1
        )  # (B, 2+k, d)

        # Step 2: key_padding_mask を作る: 先頭2つ（CLS, 関節角）は False、残りは action_is_pad → (B, 2+k)
        #         埋め草の行動を無視させないと、「エピソードの終わり」という情報が z に漏れる
        key_padding_mask = torch.cat(
            [
                torch.zeros(batch_size, 2, dtype=torch.bool, device=action_is_pad.device),
                action_is_pad,
            ],
            dim=1,
        )  # (B, 2+k)

        # Step 3: self.vae_encoder に通す（pos = self.vae_pos）
        vae_out = self.vae_encoder(
            input_tokens, pos=self.vae_pos, key_padding_mask=key_padding_mask
        )  # (B, 2+k, d)

        # Step 4: 出力の先頭（CLS の位置）だけを取り出して self.latent_proj に通し、
        #         最後の軸を半分に分けて μ と log σ² にする（Tensor.chunk(2, dim=-1)）
        mu, logvar = self.latent_proj(vae_out[:, 0, :]).chunk(2, dim=-1)  # 各 (B, latent_dim)

        return mu, logvar

    def _decode(self, image: torch.Tensor, state_n: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """方策本体: 観測と z から、正規化済みの行動チャンク (B, k, 14) を出す。

        Args:
            image: (B, 3, H, W) uint8
            state_n: (B, 14) 正規化済みの関節角。
            z: (B, latent_dim)
        Returns:
            action_chunk_n: (B, k, 14) 正規化済みの行動チャンク（予測）。
        """
        # Step 1: 画像を self.backbone に通し、self.trunk で観測トークン列と位置を得る
        feature = self.backbone(image)  # (B, 512, h, w)
        tokens, pos = self.trunk(feature, state_n)  # tokens: (B, 1+h*w, d), pos: (1, 1+h*w, d)

        # Step 2: z を射影して z トークン (B, 1, d) にし、観測トークン列の先頭に連結する
        #         位置も同じ順に連結する（self.latent_pos と trunk の pos）
        #         → トークン列 [z, 関節角, 画像 h*w 個]
        z_token = self.latent_in_proj(z).unsqueeze(1)  # (B, 1, d)
        tokens = torch.cat([z_token, tokens], dim=1)  # (B, 1+1+h*w, d)
        pos = torch.cat([self.latent_pos, pos], dim=1)  # (1, 1+1+h*w, d)

        # Step 3: self.encoder に通して memory にする
        memory = self.encoder(tokens, pos=pos)  # (B, 1+1+h*w, d)

        # Step 4: Decoder の入力 x は全て 0 のテンソル (B, k, d) にする。中身は空で、
        #         行動クエリの位置ベクトル（self.query_pos）だけで「何ステップ目の行動か」を区別する
        #         （DETR の object query と同じ考え方）。device・dtype は memory に合わせる（torch.zeros_like 等）
        batch_size, _, d_model = memory.shape
        decoder_input = torch.zeros(
            batch_size, self.chunk_size, d_model, device=memory.device, dtype=memory.dtype
        )

        # Step 5: self.decoder に通す（pos = self.query_pos、memory_pos = Step 2 の位置）
        decoder_output = self.decoder(
            decoder_input, memory, pos=self.query_pos, memory_pos=pos
        )  # (B, k, d)

        # Step 6: self.action_head をかけて (B, k, 14) にする
        action_chunk_n = self.action_head(decoder_output)  # (B, k, 14)

        return action_chunk_n

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # Step 1: 関節角と行動を正規化する（self.state_norm.normalize / self.action_norm.normalize）
        state_n = self.state_norm.normalize(batch["state"])  # (B, 14)
        action_n = self.action_norm.normalize(batch["action"])  # (B, k, 14)
        action_is_pad = batch["action_is_pad"]  # (B, k)

        # Step 2: _encode_latent で μ, log σ² を求め、reparameterize で z をサンプリングする
        mu, logvar = self._encode_latent(state_n, action_n, action_is_pad)  # 各 (B, latent_dim)
        z = reparameterize(mu, logvar)  # (B, latent_dim)

        # Step 3: _decode で行動チャンクを予測する
        action_chunk_pred_norm = self._decode(batch["image"], state_n, z)  # (B, k, 14)

        # Step 4: l1 = masked_l1(予測, 正解, batch["action_is_pad"])、kl = kl_to_standard_normal(μ, log σ²)
        #         loss = l1 + self.kl_weight * kl
        l1 = masked_l1(action_chunk_pred_norm, action_n, action_is_pad)  # スカラー
        kl = kl_to_standard_normal(mu, logvar)  # スカラー
        loss = l1 + self.kl_weight * kl  # スカラー

        # Step 5: {"loss": loss, "l1": ..., "kl": ...} を返す（ログ用の l1, kl は .detach() する）
        #         detach は計算グラフから切り離すだけで、device は変わらない（CPU へ移すのは .cpu()）。
        #         ログ用の値がグラフにつながったまま保持されると、中間の活性値まで解放されずメモリを圧迫し、
        #         誤ってそこから逆伝播する事故も起こりうる。逆伝播に使う loss だけは切り離さない
        return {"loss": loss, "l1": l1.detach(), "kl": kl.detach()}

    @torch.no_grad()
    def predict_chunk(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        # Step 1: 関節角を正規化する
        state_n = self.state_norm.normalize(obs["state"])  # (B, 14)

        # Step 2: z = 0 を作る。形 (B, latent_dim)、device・dtype は関節角に合わせる
        batch_size = state_n.shape[0]
        z = torch.zeros(
            batch_size, self.latent_dim, device=state_n.device, dtype=state_n.dtype
        )  # (B, latent_dim)

        # Step 3: _decode して、行動を逆正規化（self.action_norm.denormalize）して返す
        action_chunk_pred_norm = self._decode(obs["image"], state_n, z)  # (B, k, 14)
        action_chunk_pred = self.action_norm.denormalize(action_chunk_pred_norm)  # (B, k, 14)

        return action_chunk_pred
