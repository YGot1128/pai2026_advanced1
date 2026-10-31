"""Diffusion ヘッド: DDPM で学習し、DDPM / DDIM の2通りでサンプリングする。

考え方（Ho et al. 2020 "Denoising Diffusion Probabilistic Models"）:
    正解の行動チャンク x_0 に、少しずつノイズを足していくと、T ステップ後にはほぼ純粋なノイズになる
    （forward process）。その逆、つまり「ノイズを少しずつ除く」操作をネットワークに学習させれば、
    純粋なノイズ x_T ~ N(0, I) から出発して、観測に合った行動チャンクを作り出せる（reverse process）。

記号（t = 0, 1, ..., T-1。このファイルでは 0 始まりの添字で書く）:
    β_t:  ステップ t で足すノイズの分散（ノイズスケジュール）
    α_t = 1 - β_t
    ᾱ_t = α_0 · α_1 · ... · α_t      （累積積。x_0 の成分がどれだけ残っているか）

    forward process は、何ステップ分でも1回で書ける:
        x_t = √ᾱ_t · x_0 + √(1 - ᾱ_t) · ε,   ε ~ N(0, I)

【学習】 Ho et al. 2020 Algorithm 1
    t を {0, ..., T-1} から一様ランダムに選び、ε を N(0, I) に従う乱数で生成して、上の式で x_t を作る。
    デノイザに x_t と t を渡して ε を当てさせ、損失は MSE(ε̂, ε)（埋め草のステップは除く）。
    式で書くと（観測による条件付けは省略。θ はデノイザの重み）:
        L = E_{t, x_0, ε} ‖ ε - ε_θ(√ᾱ_t · x_0 + √(1 - ᾱ_t) · ε, t) ‖²
    括弧の中は x_t を展開したもので、E は t・x_0・ε をランダムに選んだときの平均
    （compute_loss ではバッチの平均で近似する）。‖·‖² は要素の2乗の和だが、コードでは平均をとる。
    違いは定数倍だけなので、損失を最小にする重みは同じ。
    スケジュールの値は ᾱ_t しか出てこない（β_t や、t-1 から t への進み方は出てこない）。
    そのため、ᾱ だけで書ける DDIM の更新にも、学習し直さずに同じデノイザを使える。
    ε が分かれば x_0 も分かる（x_0 = (x_t - √(1 - ᾱ_t) · ε) / √ᾱ_t）ので、ε を当てることは
    x_0 を当てることと同じ情報を持つ。

【推論 1: DDPM】 Ho et al. 2020 Algorithm 2。t = T-1, ..., 0 の T ステップを1つずつ戻る
    x_{t-1} = 1/√α_t · (x_t - β_t / √(1 - ᾱ_t) · ε̂) + σ_t · z,   z ~ N(0, I)
    σ_t² = β̃_t = (1 - ᾱ_{t-1}) / (1 - ᾱ_t) · β_t      （ᾱ_{-1} = 1 とする）
    第1項が x_{t-1} の分布の平均 μ、σ_t² がその分散で、x_{t-1} ~ N(μ, σ_t² I) から1つ取り出している。
    μ と β̃_t は、x_0 が分かっているときの x_{t-1} の分布 q(x_{t-1} | x_t, x_0)（ガウス分布）の
    平均と分散で、平均の式の x_0 に x̂_0 = (x_t - √(1 - ᾱ_t) · ε̂) / √ᾱ_t を代入すると上の形になる。
    最後のステップ（t = 0）では乱数を足さない（σ_0 = 0 になるので、足しても足さなくても同じ）。
    途中で乱数を足すので確率的: 同じ観測・同じ x_T でも、毎回違う結果になりうる。

【推論 2: DDIM（η = 0）】 Song et al. 2021 "Denoising Diffusion Implicit Models" 式 (12)
    学習は DDPM と全く同じで、辿り方だけが違う。T ステップのうち S 個だけを選んで辿る。
    t_i = T-1 - i · (T/S)   (i = 0, ..., S-1)  例: T = 100, S = 10 なら 99, 89, ..., 9
    各ステップで x_0 を推定し、ノイズ ε̂ はそのままにして、1つ前の時刻 t' の ᾱ で組み立て直す:
        x̂_0 = (x_t - √(1 - ᾱ_t) · ε̂) / √ᾱ_t
        x_{t'} = √ᾱ_{t'} · x̂_0 + √(1 - ᾱ_{t'}) · ε̂     （t' = t - T/S。最後は t' < 0 で ᾱ_{t'} = 1 とする）
    乱数を足さないので決定的: x_T を決めれば結果が決まる（微分方程式を数値的に解いているのと同じ）。

【推論 2': DDIM（η > 0）】 Song et al. 2021 式 (12)・(16)
    η = 0 の更新に、毎ステップ乱数を足す項を加える。足した乱数の分だけ ε̂ の項を小さくして、
    x_{t'} のノイズの分散を 1 - ᾱ_{t'} に保つ（学習時の x_{t'} と同じ大きさのノイズが乗った状態にする）:
        σ = η · √((1 - ᾱ_{t'}) / (1 - ᾱ_t)) · √(1 - ᾱ_t / ᾱ_{t'})
        x_{t'} = √ᾱ_{t'} · x̂_0 + √(1 - ᾱ_{t'} - σ²) · ε̂ + σ · z,   z ~ N(0, I)
    η = 0 なら σ = 0 で【推論 2】に戻る。η = 1 かつ S = T（t' = t - 1）なら σ² = β̃_t になり、
    x̂_0 がクリップされない限り DDPM（【推論 1】）と同じ更新になる。0 < η < 1 はその中間。
    最後のステップ（ᾱ_{t'} = 1）では σ = 0 なので、乱数は足されない。
    η は 0〜1 で使う（η > 1 では 1 - ᾱ_{t'} - σ² が負になりうる）。

【x̂_0 のクリップ】 Diffusion Policy（Chi et al. 2023）が使う diffusers の clip_sample=True と同じ考え方
    ε̂ の誤差を e（ε̂ = ε + e）とすると、x̂_0 = x_0 - √(1 - ᾱ_t) / √ᾱ_t · e となり、誤差が
    √(1 - ᾱ_t) / √ᾱ_t 倍されて x̂_0 に乗る。cosine スケジュールでは ᾱ_{T-1} ≈ 2.4e-7 なので、
    最初のステップ（t = T-1）ではこの倍率が約 2000 になる。300 ステップの試走のモデル（ε̂ の誤差 0.28）では
    x̂_0 が正規化後の単位で約 570 にずれ、シミュレータが発散した（DIFFUSION_IMPLEMENTATION.md の D7b）。
    そこで各ステップの x̂_0 を [-x0_clip, x0_clip] にクリップしてから次の x を作る
    （クリップ: 範囲からはみ出た要素を端の値に置き換え、範囲内の要素はそのまま残す）。
    学習データの行動は正規化後の単位で最大 5.2 なので、既定の x0_clip = 6 では正しい x̂_0 の値はほとんど変わらない。
    DDPM でもクリップできるように、平均 μ を x̂_0 を使う形（Ho et al. 2020 式 (7) の事後分布の平均）で書く:
        μ = √ᾱ_{t-1} · β_t / (1 - ᾱ_t) · x̂_0 + √α_t · (1 - ᾱ_{t-1}) / (1 - ᾱ_t) · x_t     （ᾱ_{-1} = 1）
    クリップしなければ、x̂_0 を代入して整理すると【推論 1】の平均の式と一致する。

ノイズスケジュール（Nichol & Dhariwal 2021 "Improved DDPM" Sec. 3.2 の cosine スケジュール）:
    f(u) = cos²((u + s) / (1 + s) · π/2),  s = 0.008
    β_t = min(1 - f((t+1)/T) / f(t/T), 0.999)
    ᾱ_t が 1 から 0 までなめらかに減る。T = 100 のような少ないステップ数でも、
    最後にはほぼ純粋なノイズになる（DDPM 原著の線形スケジュールは T = 1000 用）。

    u = t/T は時刻を 0〜1 に直したもの。f(u) は「時刻 u で x_0 の成分がどれだけ残っているか」を表す。
    s は t = 0 付近で β が小さくなりすぎないためのずらし（s = 0 だと β_0 = 2.5e-4。s = 0.008 では 6.3e-4）。

    ᾱ_t は f の比で直接書ける。1 - β_t = f((t+1)/T) / f(t/T) を掛け合わせると、途中が打ち消し合う:
        ᾱ_t = (1 - β_0) · (1 - β_1) · ... · (1 - β_t)
            = f(1/T)/f(0) · f(2/T)/f(1/T) · ... · f((t+1)/T)/f(t/T)    … 1 - β の式を代入
            = f((t+1)/T) / f(0)                                         … 隣どうしの分子と分母が消える
    これは β が上限 0.999 に当たらない t で成り立つ。T = 100 で上限に当たるのは t = 99 の1個だけ:
        f(1) = cos²((1 + s) / (1 + s) · π/2) = cos²(π/2) = 0             … u = 1 を代入
        β_99 = min(1 - 0 / f(0.99), 0.999) = min(1, 0.999) = 0.999       … 上限がないと β = 1、ᾱ_99 = 0 になる
        ᾱ_99 = ᾱ_98 · (1 - β_99) = 2.43e-4 × 0.001 = 2.43e-7             … 0 にならず、小さな値が残る

    よく使う値（T = 100。cosine_beta_schedule(100) から計算）:
        t  | β_t      | ᾱ_t      | √ᾱ_t    | √(1 - ᾱ_t) | √(1 - ᾱ_t) / √ᾱ_t
        0  | 0.00063  | 0.9994   | 0.9997  | 0.0251     | 0.0251
        1  | 0.0011   | 0.9983   | 0.9991  | 0.0418     | 0.0418
        9  | 0.0050   | 0.972    | 0.986   | 0.167      | 0.169
        49 | 0.0306   | 0.494    | 0.703   | 0.711      | 1.01
        50 | 0.0315   | 0.478    | 0.692   | 0.722      | 1.04
        59 | 0.0418   | 0.341    | 0.584   | 0.812      | 1.39
        89 | 0.172    | 0.0241   | 0.155   | 0.988      | 6.37
        98 | 0.750    | 2.43e-4  | 0.0156  | 1.000      | 64.2
        99 | 0.999    | 2.43e-7  | 4.9e-4  | 1.000      | 2029
    √ᾱ_t は x_t の中の x_0 の係数、√(1 - ᾱ_t) はノイズの係数。最後の列は、x̂_0 を求めるときに
    ε̂ の誤差が何倍されるか（【x̂_0 のクリップ】）。t = 99 で約 2000 倍になる。
    DDIM 10 は t = 99, 89, ..., 9 を、DDIM 100 と DDPM は t = 99, 98, ..., 0 を辿る。

ハイパーパラメータの既定値は Diffusion Policy（Chi et al. 2023）に合わせる:
    T 100 / cosine スケジュール / ε を予測 / DDIM の推論 10 ステップ
ネットワークの大きさ（d_model など）は ACT と同じ（policies/denoiser.py）。
"""

import math

import torch

from pai_act.policies.base import ACTION_DIM
from pai_act.policies.denoiser import DenoiserPolicy

SAMPLERS = ("ddpm", "ddim")


def cosine_beta_schedule(num_steps: int, s: float = 0.008, max_beta: float = 0.999) -> torch.Tensor:
    """cosine スケジュールの β_t (t = 0, ..., T-1) を作る（式はモジュール docstring）。

    Args:
        num_steps: 拡散のステップ数 T。
        s: t = 0 付近で β が小さくなりすぎないためのずらし。
        max_beta: β の上限。t = T-1 では f(1) = 0 となり β = 1 になってしまうので抑える。

    Returns:
        (T,) float32
    """
    # Step 1: u = 0, 1/T, 2/T, ..., 1 の T+1 個を作る（torch.arange を T で割る。float64 で計算すると安心）
    u = torch.arange(0, num_steps + 1, dtype=torch.float64) / num_steps  # (T+1,)

    # Step 2: f(u) = cos²((u + s) / (1 + s) · π/2) を計算する → (T+1,)
    f = torch.cos((u + s) / (1 + s) * math.pi / 2) ** 2  # (T+1,)

    # Step 3: β_t = 1 - f[t+1] / f[t] を T 個作り（スライスで1つずらして割る）、上限 max_beta で抑える
    #         （Tensor.clamp(max=...)）。float32 にして返す
    betas = 1 - f[1:] / f[:-1]  # (T,)
    betas = betas.clamp(max=max_beta).to(torch.float32)  # (T,)

    return betas


class DiffusionPolicy(DenoiserPolicy):
    """Diffusion ヘッド。ネットワークの引数は DenoiserPolicy を参照。

    Args:
        num_diffusion_steps: 拡散のステップ数 T。
        sampler: predict_chunk で使うサンプラー（"ddpm" / "ddim"）。評価時に上書きできる。
        num_inference_steps: DDIM で辿るステップ数 S（T の約数）。評価時に上書きできる。
        x0_clip: サンプリング中の x̂_0 を [-x0_clip, x0_clip] にクリップする
            （正規化後の単位。0 でクリップしない）。
            評価時に上書きできる。
        ddim_eta: DDIM で毎ステップ足す乱数の大きさ η（0〜1。0 で決定的、1 かつ S = T で DDPM と同じ）。
            評価時に上書きできる。
    """

    # register_buffer で作る属性は、型チェッカーに Tensor だと伝えるために宣言しておく
    betas: torch.Tensor
    alphas: torch.Tensor
    alphas_cumprod: torch.Tensor

    def __init__(
        self,
        chunk_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        dim_feedforward: int = 3200,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 1,
        dropout: float = 0.1,
        num_diffusion_steps: int = 100,
        sampler: str = "ddim",
        num_inference_steps: int = 10,
        x0_clip: float = 6.0,
        ddim_eta: float = 0.0,
        pretrained_backbone: bool = True,
    ):
        super().__init__(
            chunk_size,
            d_model=d_model,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout,
            pretrained_backbone=pretrained_backbone,
        )
        self.num_diffusion_steps = num_diffusion_steps
        self.sampler = sampler
        self.num_inference_steps = num_inference_steps
        self.x0_clip = x0_clip
        self.ddim_eta = ddim_eta

        # Step 1: betas = cosine_beta_schedule(num_diffusion_steps)、alphas = 1 - betas、
        #         alphas_cumprod = alphas の累積積（torch.cumprod）を作る。いずれも (T,)
        betas = cosine_beta_schedule(num_diffusion_steps)  # (T,)
        alphas = 1 - betas  # (T,)
        alphas_cumprod = torch.cumprod(alphas, dim=0)  # (T,)

        # Step 2: 3つを self.register_buffer で持つ（名前は betas / alphas / alphas_cumprod）
        #         スケジュールは T から決まる値なので persistent=False でよい（ACT の vae_pos と同じ）
        self.register_buffer("betas", betas, persistent=False)
        self.register_buffer("alphas", alphas, persistent=False)
        self.register_buffer("alphas_cumprod", alphas_cumprod, persistent=False)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """forward process: x_0 に t ステップ分のノイズを1回で乗せる。

        Args:
            x0: (B, k, 14) 正規化済みの行動チャンク。
            t: (B,) long。サンプルごとに違ってよい。
            noise: (B, k, 14) ε ~ N(0, I)。

        Returns:
            x_t: (B, k, 14)
        """
        # Step 1: self.alphas_cumprod[t] で各サンプルの ᾱ_t (B,) を取り出し、(B, 1, 1) にする
        #         （x0 の (B, k, 14) とブロードキャストできる形）
        alphas_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1)  # (B, 1, 1)

        # Step 2: x_t = √ᾱ_t · x0 + √(1 - ᾱ_t) · noise を返す
        x_t = (
            torch.sqrt(alphas_cumprod_t) * x0 + torch.sqrt(1 - alphas_cumprod_t) * noise
        )  # (B, k, 14)

        return x_t

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # Step 1: 関節角と行動を正規化する（self.state_norm.normalize / self.action_norm.normalize）
        state, action = batch["state"], batch["action"]  # (B, 14), (B, k, 14)
        state_norm = self.state_norm.normalize(state)  # (B, 14)
        action_norm = self.action_norm.normalize(action)  # (B, k, 14)
        image = batch["image"]  # (B, 3, H, W) uint8

        # Step 2: self._encode_obs で memory と memory_pos を作る
        # shapeのLは、画像トークンの数 h*w + 1（関節角のトークン）で、k ではないことに注意
        memory, memory_pos = self._encode_obs(image, state_norm)  # (B, L, d_model), (1, L, d_model)

        # Step 3: t をバッチ内のサンプルごとに {0, ..., T-1} から一様ランダムに選ぶ:
        #         torch.randint → (B,)
        #         ノイズ ε を正規化した行動と同じ形で、標準正規分布の乱数として生成する:
        #         torch.randn_like → (B, k, 14)
        #         （device は行動に合わせる）
        batch_size = action_norm.shape[0]
        t = torch.randint(
            0, self.num_diffusion_steps, (batch_size,), device=action_norm.device
        )  # (B,)
        noise = torch.randn_like(action_norm, device=action_norm.device)  # (B, k, 14)

        # Step 4: self.q_sample で x_t を作り、self._denoise で ε を予測する
        x_t = self.q_sample(action_norm, t, noise)  # (B, k, 14)
        epsilon_hat = self._denoise(x_t, t, memory, memory_pos)  # (B, k, 14)

        # Step 5: 埋め草を除いた MSE: ステップごとに 14 次元の平均を取り (B, k)、
        #         batch["action_is_pad"] が False のステップだけで平均する（policies/base.py の
        #         masked_l1 と同じ集約で、絶対値の代わりに2乗を使う）
        #         返り値は {"loss": mse, "mse": mse.detach()}
        action_is_pad = batch["action_is_pad"]  # (B, k) bool
        mse = ((epsilon_hat - noise) ** 2).mean(dim=-1)  # (B, k)
        mse = mse.masked_fill(action_is_pad, 0.0)  # 埋め草のステップは0にする
        mse = mse.sum() / (~action_is_pad).sum()  # 埋め草以外のステップで平均する

        return {"loss": mse, "mse": mse.detach()}

    def _sample_ddpm(
        self, x: torch.Tensor, memory: torch.Tensor, memory_pos: torch.Tensor
    ) -> torch.Tensor:
        """DDPM: x_T から t = T-1, ..., 0 の T ステップを1つずつ戻って x_0 を作る。

        Args:
            x: (B, k, 14) 出発点 x_T ~ N(0, I)。
            memory, memory_pos: _encode_obs の出力。

        Returns:
            (B, k, 14) 正規化済みの行動チャンク。
        """
        batch_size = x.shape[0]
        # Step 1: t = T-1, ..., 0 の順にループする（reversed(range(...))）
        for t in reversed(range(self.num_diffusion_steps)):
            # Step 2: t をバッチの数だけ並べた (B,) の long テンソルを作り、self._denoise で ε̂ を予測する
            #         （torch.full。device は x に合わせる）
            t_batch = torch.full((batch_size,), t, dtype=torch.long, device=x.device)  # (B,)
            epsilon_hat = self._denoise(x, t_batch, memory, memory_pos)  # (B, k, 14)

            # Step 3（D7b で書き換える）: x̂_0 をクリップできるように、平均を x̂_0 を使う形にする
            #   3a: x̂_0 = (x_t - √(1 - ᾱ_t) · ε̂) / √ᾱ_t を求める（DDIM の Step 2b と同じ式）
            x_hat_0 = (x - torch.sqrt(1 - self.alphas_cumprod[t]) * epsilon_hat) / torch.sqrt(
                self.alphas_cumprod[t]
            )  # (B, k, 14)

            #   3b: self.x0_clip > 0 なら x̂_0 を [-x0_clip, x0_clip] にクリップする（torch.clamp）
            if self.x0_clip > 0:
                x_hat_0 = torch.clamp(x_hat_0, min=-self.x0_clip, max=self.x0_clip)

            #   3c: ᾱ_{t-1} を取り出す。t = 0 なら 1 にする（DDIM の Step 2c と同じ扱い）
            if t > 0:
                alphas_cumprod_prev = self.alphas_cumprod[t - 1]
            else:
                alphas_cumprod_prev = self.alphas_cumprod.new_tensor(1.0)

            #   3d: 平均 μ = √ᾱ_{t-1} · β_t / (1 - ᾱ_t) · x̂_0 + √α_t · (1 - ᾱ_{t-1}) / (1 - ᾱ_t) · x_t
            #       t = 0 では x̂_0 の係数が 1、x_t の係数が 0 になり、μ = x̂_0（クリップした値）そのものになる
            mean = (
                torch.sqrt(alphas_cumprod_prev)
                * self.betas[t]
                / (1 - self.alphas_cumprod[t])
                * x_hat_0
                + torch.sqrt(self.alphas[t])
                * (1 - alphas_cumprod_prev)
                / (1 - self.alphas_cumprod[t])
                * x
            )  # (B, k, 14)

            # Step 4: t > 0 なら σ_t · z を足す（σ_t² = β̃_t、z は torch.randn_like）。t = 0 なら平均をそのまま使う
            if t > 0:
                beta_tilde = (
                    (1 - alphas_cumprod_prev) / (1 - self.alphas_cumprod[t]) * self.betas[t]
                )
                sigma = torch.sqrt(beta_tilde)
                z = torch.randn_like(x)  # (B, k, 14)
                x = mean + sigma * z
            else:
                x = mean

        return x

    def _sample_ddim(
        self, x: torch.Tensor, memory: torch.Tensor, memory_pos: torch.Tensor, num_steps: int
    ) -> torch.Tensor:
        """DDIM: T ステップのうち num_steps 個を辿って x_0 を作る。

        self.ddim_eta = 0 なら乱数を使わない（決定的）。η > 0 なら毎ステップ σ · z を足す（D11）。

        Args:
            x: (B, k, 14) 出発点 x_T ~ N(0, I)。
            memory, memory_pos: _encode_obs の出力。
            num_steps: 辿るステップ数 S（T の約数）。

        Returns:
            (B, k, 14) 正規化済みの行動チャンク。
        """
        # Step 1: T が num_steps で割り切れることを assert で確かめ、刻み幅 T // num_steps を求める
        T = self.num_diffusion_steps
        assert T % num_steps == 0, f"T={T} must be divisible by num_steps={num_steps}"
        step_size = T // num_steps

        # Step 2: t = T-1 から刻み幅ずつ減らしながら、t >= 0 の間ループする（range の3つ目の引数を負にする）
        batch_size = x.shape[0]
        for t in range(T - 1, -1, -step_size):
            # Step 2a: t をバッチの数だけ並べた (B,) の long テンソルを作り、self._denoise で ε̂ を予測する
            #          （torch.full。device は x に合わせる）
            t_batch = torch.full((batch_size,), t, dtype=torch.long, device=x.device)  # (B,)
            epsilon_hat = self._denoise(x, t_batch, memory, memory_pos)  # (B, k, 14)

            # Step 2b: モジュール docstring の式で x̂_0 を推定する: (x_t - √(1 - ᾱ_t) · ε̂) / √ᾱ_t
            x_hat_0 = (x - torch.sqrt(1 - self.alphas_cumprod[t]) * epsilon_hat) / torch.sqrt(
                self.alphas_cumprod[t]
            )  # (B, k, 14)
            # Step 2b'（D7b で追加）: self.x0_clip > 0 なら x̂_0 を [-x0_clip, x0_clip] にクリップする
            #          （torch.clamp）。ε̂ はクリップした x̂_0 から計算し直さず、2d でそのまま使う
            #          （diffusers の DDIM と同じ扱い）
            if self.x0_clip > 0:
                x_hat_0 = torch.clamp(x_hat_0, min=-self.x0_clip, max=self.x0_clip)

            # Step 2c: t' = t - 刻み幅 の ᾱ_{t'} を取り出す。t' >= 0 なら self.alphas_cumprod[t']、
            #          t' < 0（最後のステップ）なら 1 にする（if で分ける。t は Python の int）
            #          t' を 0 に切り上げると ᾱ_0 ≈ 0.9994 でノイズが残り、添字 -1 で引くと末尾の
            #          ᾱ_{T-1} ≈ 0 になるので、どちらも誤り。1 もテンソルで持つと 2d の torch.sqrt に
            #          そのまま渡せる（self.alphas_cumprod.new_tensor(1.0) で device と dtype が揃う）
            t_prime = t - step_size
            if t_prime >= 0:
                alpha_prime = self.alphas_cumprod[t_prime]
            else:
                alpha_prime = self.alphas_cumprod.new_tensor(1.0)

            # Step 2d: モジュール docstring の式で x_{t'} を作る: √ᾱ_{t'} · x̂_0 + √(1 - ᾱ_{t'}) · ε̂
            #   （D11 で書き換える）【推論 2'】の式にして η > 0 に対応させる。η = 0 では今と同じ値になる:
            #   2d-1: σ = η · √((1 - ᾱ_{t'}) / (1 - ᾱ_t)) · √(1 - ᾱ_t / ᾱ_{t'}) を求める（η は self.ddim_eta）。
            #         最後のステップ（ᾱ_{t'} = 1）では σ = 0 になる
            sigma = (
                self.ddim_eta
                * torch.sqrt((1 - alpha_prime) / (1 - self.alphas_cumprod[t]))
                * torch.sqrt(1 - self.alphas_cumprod[t] / alpha_prime)
            )  # スカラ値

            #   2d-2: x_{t'} = √ᾱ_{t'} · x̂_0 + √(1 - ᾱ_{t'} - σ²) · ε̂ + σ · z（z は torch.randn_like）。
            #         ε̂ の係数が √(1 - ᾱ_{t'}) から √(1 - ᾱ_{t'} - σ²) に変わる（足す乱数の分だけ小さくする）
            x = (
                torch.sqrt(alpha_prime) * x_hat_0
                + torch.sqrt(1 - alpha_prime - sigma**2) * epsilon_hat
                + sigma * torch.randn_like(x, dtype=x.dtype, device=x.device)
            )  # (B, k, 14)

        # Step 3: ループを抜けた x を返す。最後のステップ（t' < 0）で ᾱ_{t'} = 1 としたので、
        #         x は x̂_0 そのものになっている。デノイザを呼ぶのはループの中の S 回だけ
        return x

    @torch.no_grad()
    def predict_chunk(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        # Step 1: 関節角を正規化し、self._encode_obs で memory を作る（サンプリングの前に1回だけ）
        state = obs["state"]  # (B, 14)
        state_norm = self.state_norm.normalize(state)  # (B, 14)
        image = obs["image"]  # (B, 3, H, W) uint8

        # shapeのLは、画像トークンの数 h*w + 1（関節角のトークン）で、k ではないことに注意
        memory, memory_pos = self._encode_obs(image, state_norm)  # (B, L, d_model), (1, L, d_model)

        # Step 2: 出発点 x_T ~ N(0, I) を (B, k, 14) の乱数として生成する
        #         （torch.randn。device は memory に合わせる）
        batch_size = state.shape[0]
        x = torch.randn(
            (batch_size, self.chunk_size, ACTION_DIM), device=memory.device
        )  # (B, k, 14)

        # Step 3: self.sampler が "ddpm" なら _sample_ddpm、"ddim" なら _sample_ddim
        #         （ステップ数は self.num_inference_steps）、それ以外は ValueError
        if self.sampler == "ddpm":
            action_chunk_pred_norm = self._sample_ddpm(x, memory, memory_pos)
        elif self.sampler == "ddim":
            action_chunk_pred_norm = self._sample_ddim(
                x, memory, memory_pos, self.num_inference_steps
            )
        else:
            raise ValueError(f"unknown sampler: {self.sampler} (expected one of {SAMPLERS})")

        # Step 4: 行動の正規化を戻して（self.action_norm.denormalize）返す
        action_chunk_pred = self.action_norm.denormalize(action_chunk_pred_norm)  # (B, k, 14)

        return action_chunk_pred
