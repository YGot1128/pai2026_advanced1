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

ノイズスケジュール（Nichol & Dhariwal 2021 "Improved DDPM" Sec. 3.2 の cosine スケジュール）:
    f(u) = cos²((u + s) / (1 + s) · π/2),  s = 0.008
    β_t = min(1 - f((t+1)/T) / f(t/T), 0.999)
    ᾱ_t が 1 から 0 までなめらかに減る。T = 100 のような少ないステップ数でも、
    最後にはほぼ純粋なノイズになる（DDPM 原著の線形スケジュールは T = 1000 用）。

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
        # shapeのLは、画像のパッチ数 + 1（cls token）で、k ではないことに注意
        memory, memory_pos = self._encode_obs(image, state_norm)  # (B, L, d_model), (B, L, d_model)

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
        x_t = self.q_sample(action_norm, t, noise)
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

            # Step 3: モジュール docstring の式で平均を作る: 1/√α_t · (x_t - β_t / √(1 - ᾱ_t) · ε̂)
            mean = (
                1
                / torch.sqrt(self.alphas[t])
                * (x - self.betas[t] / torch.sqrt(1 - self.alphas_cumprod[t]) * epsilon_hat)
            )

            # Step 4: t > 0 なら σ_t · z を足す（σ_t² = β̃_t、z は torch.randn_like）。t = 0 なら平均をそのまま使う
            if t > 0:
                beta_tilde = (
                    (1 - self.alphas_cumprod[t - 1]) / (1 - self.alphas_cumprod[t]) * self.betas[t]
                )
                sigma = torch.sqrt(beta_tilde)
                z = torch.randn_like(x)
                x = mean + sigma * z
            else:
                x = mean

        return x

    def _sample_ddim(
        self, x: torch.Tensor, memory: torch.Tensor, memory_pos: torch.Tensor, num_steps: int
    ) -> torch.Tensor:
        """DDIM（η = 0）: T ステップのうち num_steps 個を辿って x_0 を作る。乱数は使わない。

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
            )

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
            x = torch.sqrt(alpha_prime) * x_hat_0 + torch.sqrt(1 - alpha_prime) * epsilon_hat

        # Step 3: ループを抜けた x を返す。最後のステップ（t' < 0）で ᾱ_{t'} = 1 としたので、
        #         x は x̂_0 そのものになっている。デノイザを呼ぶのはループの中の S 回だけ
        return x

    @torch.no_grad()
    def predict_chunk(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        # Step 1: 関節角を正規化し、self._encode_obs で memory を作る（サンプリングの前に1回だけ）
        state = obs["state"]  # (B, 14)
        state_norm = self.state_norm.normalize(state)  # (B, 14)
        image = obs["image"]

        # shapeのLは、画像のパッチ数 + 1（cls token）で、k ではないことに注意
        memory, memory_pos = self._encode_obs(image, state_norm)  # (B, L, d_model), (B, L, d_model)

        # Step 2: 出発点 x_T ~ N(0, I) を (B, k, 14) の乱数として生成する
        #         （torch.randn。device は memory に合わせる）
        batch_size = state.shape[0]
        x = torch.randn((batch_size, self.chunk_size, ACTION_DIM), device=memory.device)

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
