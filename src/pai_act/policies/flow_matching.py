"""Flow Matching ヘッド: ノイズから行動へ向かう速度を学習し、ODE を Euler / Heun で解いて行動を作る。

考え方（Lipman et al. 2023 "Flow Matching for Generative Modeling" / Liu et al. 2023 "Flow Straight and Fast"
（Rectified Flow））:
    正解の行動チャンク x_0 とノイズ ε を直線で結び、その直線の上を動く点の速度をネットワークに当てさせる。
    （同じ点を通る直線はたくさんあるので、実際に学習されるのはそれらの速度の平均。【学習】を参照）
    推論では、純粋なノイズから出発して、学習した速度に沿って少しずつ動かすと、観測に合った行動チャンクにたどり着く。
    Diffusion（policies/diffusion.py）と違い、ノイズスケジュール（β_t, ᾱ_t）がない。

記号（t は 0〜1 の実数。Diffusion と同じく、t が大きいほどノイズが多い）:
    x_t = (1 - t) · x_0 + t · ε,   ε ~ N(0, I)      t = 0 で x_0（行動）、t = 1 で ε（純粋なノイズ）
    u   = dx_t/dt = ε - x_0                          x_t を t で微分したもの（1本の直線の上では t によらず一定）
    v̂ = v_θ(x_t, t)                                  ネットワーク v_θ（θ は重み）が出す速度の予測。x_t と t（と観測）で変わる
    t の向きは openpi（π0 の公式実装）・LeRobot の pi0 と同じ（π0 の論文と Lipman et al. は逆向きの t で書いている）。

【学習】 conditional flow matching（Lipman et al. 2023。Rectified Flow と同じ直線のパス）
    t を [0, 1) から一様ランダムに選び、ε を N(0, I) に従う乱数で生成して、上の式で x_t を作る。
    デノイザに x_t と t を渡して u = ε - x_0 を当てさせ、損失は MSE(v̂, u)（埋め草のステップは除く）。
        L = E_{t, x_0, ε} ‖ v_θ((1 - t) · x_0 + t · ε, t) - (ε - x_0) ‖²
    同じ x_t を通る直線は、(x_0, ε) の組を変えるとたくさんある。そのため u は x_t からは1つに決まらず、
    MSE を最小にする v̂ は「x_t を通る直線の速度の平均」E[ε - x_0 | x_t, t, 観測] になる。
    この平均の速度場（各点 x と時刻 t に速度を1つ決めたもの）に沿って動かすと、t = 1 の N(0, I) が
    t = 0 のデータの分布に運ばれる（Lipman et al. 2023 の定理）。

【推論】 x_1 ~ N(0, I) から、ODE dx/dt = v̂(x, t) を t = 1 → 0 の向きに N ステップで解く（刻み幅 Δ = 1/N）。
    t が減る向きに進むので、どちらのソルバーも「x から Δ · 速度 を引く」形になる。
    Euler（1次）: 傾きをいまの点で1回だけ測って進む。デノイザを1ステップに1回呼ぶ
        x_{t-Δ} = x_t - Δ · v̂(x_t, t)
    Heun（2次）: Euler で仮に進んだ先でも傾きを測り、2つの傾きの平均で進み直す。デノイザを1ステップに2回呼ぶ
        v_1 = v̂(x_t, t)
        x̃   = x_t - Δ · v_1                  … Euler で仮に進んだ点
        v_2 = v̂(x̃, t - Δ)                   … 仮に進んだ先の傾き
        x_{t-Δ} = x_t - Δ · (v_1 + v_2) / 2
    デノイザを呼ぶ回数（NFE: number of function evaluations）は Euler で N 回、Heun で 2N 回。
    計算量を揃えて比べるときは、Euler N = 10 と Heun N = 5 のように NFE を合わせる。
    どちらも乱数を足さないので決定的: x_1 を決めれば結果が決まる（η = 0 の DDIM と同じ）。

Diffusion との違い:
    - 当てるもの: 足されたノイズ ε → 速度 ε - x_0
    - x_t の作り方: √ᾱ_t · x_0 + √(1 - ᾱ_t) · ε（スケジュールが要る）→ 直線 (1 - t) · x_0 + t · ε
    - x_0 の推定: x̂_0 = x_t - t · v̂ なので、v̂ の誤差は x̂_0 では t 倍（1 以下）にしかならない。
      Diffusion の t = 99 で ε̂ の誤差が約 2000 倍になる問題（DIFFUSION_IMPLEMENTATION.md の D7b）は起きないので、
      x̂_0 のクリップは置かない

時刻の埋め込み:
    DenoiserPolicy._embed_time は t を拡大縮小せずに sinusoidal にする。0〜1 のままでは周波数の低い成分が
    ほとんど変わらず t を区別しにくいので、time_scale（既定 100）を掛けてから渡す（_velocity）。
    100 にすると、Diffusion の t = 0〜99 と同じ範囲になり、時刻の埋め込みの条件が揃う。

ハイパーパラメータの既定値:
    t の分布は一様（Lipman et al. 2023）/ 推論は Euler 10 ステップ（π0 と同じ）
    ネットワークの大きさ（d_model など）は ACT・Diffusion と同じ（policies/denoiser.py）。
"""

import torch

from pai_act.policies.base import ACTION_DIM
from pai_act.policies.denoiser import DenoiserPolicy

SOLVERS = ("euler", "heun")


class FlowMatchingPolicy(DenoiserPolicy):
    """Flow Matching ヘッド。ネットワークの引数は DenoiserPolicy を参照。

    Args:
        solver: predict_chunk で使う ODE ソルバー（"euler" / "heun"）。評価時に上書きできる。
        num_inference_steps: ODE を解くステップ数 N（Heun はデノイザを 2N 回呼ぶ）。評価時に上書きできる。
        time_scale: デノイザに渡す前に t に掛ける倍率（モジュール docstring の「時刻の埋め込み」）。
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
        solver: str = "euler",
        num_inference_steps: int = 10,
        time_scale: float = 100.0,
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
        self.solver = solver
        self.num_inference_steps = num_inference_steps
        self.time_scale = time_scale
        # ノイズスケジュールがないので、Diffusion と違ってバッファは持たない

    def interpolate(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """x_0 とノイズを結ぶ直線の上の、時刻 t の点 x_t を作る（Diffusion の q_sample にあたる）。

        Args:
            x0: (B, k, 14) 正規化済みの行動チャンク。
            t: (B,) float。0〜1 の実数で、サンプルごとに違ってよい。
            noise: (B, k, 14) ε ~ N(0, I)。

        Returns:
            x_t: (B, k, 14)
        """
        # Step 1: t を (B, 1, 1) にする（x0 の (B, k, 14) とブロードキャストできる形）
        t = t.view(-1, 1, 1)  # (B, 1, 1)

        # Step 2: x_t = (1 - t) · x0 + t · noise を返す
        x_t = (1 - t) * x0 + t * noise  # (B, k, 14)

        return x_t

    def _velocity(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        memory: torch.Tensor,
        memory_pos: torch.Tensor,
    ) -> torch.Tensor:
        """x_t と時刻 t（0〜1）から速度 v̂ を予測する。学習・推論のどちらもデノイザはここから呼ぶ。

        Args:
            x_t: (B, k, 14) 直線の上の点（正規化済みの空間）。
            t: (B,) float。0〜1 の実数。
            memory, memory_pos: _encode_obs の出力。

        Returns:
            v̂: (B, k, 14)
        """
        # Step 1: t に self.time_scale を掛けて self._denoise に渡し、その出力を返す
        #         （t は整数に丸めない。Euler 10 ステップなら t = 1.0, 0.9, ... で、100 倍すると 100, 90, ...）
        velocity = self._denoise(x_t, t * self.time_scale, memory, memory_pos)  # (B, k, 14)

        return velocity

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # Step 1: 関節角と行動を正規化する（self.state_norm.normalize / self.action_norm.normalize）
        state, action = batch["state"], batch["action"]  # (B, 14), (B, k, 14)
        state_norm = self.state_norm.normalize(state)  # (B, 14)
        action_norm = self.action_norm.normalize(action)  # (B, k, 14)
        image = batch["image"]  # (B, 3, H, W) uint8

        # Step 2: self._encode_obs で memory と memory_pos を作る
        # shapeのLは、画像トークンの数 h*w + 1（関節角のトークン）で、k ではないことに注意
        memory, memory_pos = self._encode_obs(image, state_norm)  # (B, L, d_model), (1, L, d_model)

        # Step 3: t をバッチ内のサンプルごとに [0, 1) から一様ランダムに選ぶ: torch.rand → (B,)
        #         ノイズ ε を正規化した行動と同じ形で、標準正規分布の乱数として生成する: torch.randn_like
        #         （device は行動に合わせる）
        batch_size = action_norm.shape[0]
        t = torch.rand(batch_size, device=action_norm.device)  # (B,)
        noise = torch.randn_like(action_norm, device=action_norm.device)  # (B, k, 14)

        # Step 4: self.interpolate で x_t を作り、self._velocity で速度 v̂ を予測する
        #         正解の速度 u = ε - x_0（x_0 は正規化した行動）を作る
        x_t = self.interpolate(action_norm, t, noise)
        velocity_pred = self._velocity(x_t, t, memory, memory_pos)
        velocity_target = noise - action_norm

        # Step 5: 埋め草を除いた MSE（Diffusion の compute_loss の Step 5 と同じ集約。比べる相手が ε ではなく u）
        #         返り値は {"loss": mse, "mse": mse.detach()}
        action_is_pad = batch["action_is_pad"]  # (B, k) bool
        mse = ((velocity_pred - velocity_target) ** 2).mean(dim=-1)  # (B, k)
        mse = mse.masked_fill(action_is_pad, 0.0)  # 埋め草のステップは0にする
        mse = mse.sum() / (~action_is_pad).sum()  # 埋め草以外のステップで平均する

        return {"loss": mse, "mse": mse.detach()}

    def _sample_euler(
        self, x: torch.Tensor, memory: torch.Tensor, memory_pos: torch.Tensor, num_steps: int
    ) -> torch.Tensor:
        """Euler 法: t = 1 から 0 まで、num_steps 回に分けて速度に沿って進む。乱数は使わない。

        Args:
            x: (B, k, 14) 出発点 x_1 ~ N(0, I)。
            memory, memory_pos: _encode_obs の出力。
            num_steps: ステップ数 N。

        Returns:
            (B, k, 14) 正規化済みの行動チャンク（t = 0 の点）。
        """
        # Step 1: 刻み幅 Δ = 1 / num_steps を求める
        assert num_steps >= 1, f"num_steps={num_steps} must be at least 1"
        delta = 1 / num_steps

        # Step 2: i = 0, ..., num_steps - 1 でループする
        batch_size = x.shape[0]
        for i in range(num_steps):
            #   2a: いまの時刻 t = 1 - i / num_steps を、バッチの数だけ並べた (B,) の float テンソルにする
            #       （torch.full。dtype と device は x に合わせる。Δ を i 回引くより、i / num_steps のほうが誤差が少ない）
            t = 1 - i / num_steps
            t_batch = torch.full((batch_size,), t, dtype=x.dtype, device=x.device)  # (B,)

            #   2b: self._velocity で v̂ を予測し、x = x - Δ · v̂ で1ステップ進む（t が減る向きなので引く）
            velocity_pred = self._velocity(x, t_batch, memory, memory_pos)
            x = x - delta * velocity_pred

        # Step 3: ループを抜けた x を返す（デノイザを呼ぶのは num_steps 回。t = 0 では呼ばない）
        return x

    def _sample_heun(
        self, x: torch.Tensor, memory: torch.Tensor, memory_pos: torch.Tensor, num_steps: int
    ) -> torch.Tensor:
        """Heun 法: 各ステップで傾きを2回測り、その平均で進む。乱数は使わない。

        Args:
            x: (B, k, 14) 出発点 x_1 ~ N(0, I)。
            memory, memory_pos: _encode_obs の出力。
            num_steps: ステップ数 N（デノイザは 2N 回呼ぶ）。

        Returns:
            (B, k, 14) 正規化済みの行動チャンク（t = 0 の点）。
        """
        # Step 1: 刻み幅 Δ = 1 / num_steps を求める
        assert num_steps >= 1, f"num_steps={num_steps} must be at least 1"
        delta = 1 / num_steps

        # Step 2: i = 0, ..., num_steps - 1 でループする
        batch_size = x.shape[0]
        for i in range(num_steps):
            #   2a: いまの時刻 t = 1 - i / num_steps と、次の時刻 t - Δ = 1 - (i + 1) / num_steps を、
            #       それぞれ (B,) の float テンソルにする（_sample_euler の 2a と同じ作り方）
            t = 1 - i / num_steps
            t_batch = torch.full((batch_size,), t, dtype=x.dtype, device=x.device)  # (B,)
            next_t = 1 - (i + 1) / num_steps
            next_t_batch = torch.full((batch_size,), next_t, dtype=x.dtype, device=x.device)  # (B,)

            #   2b: いまの点の傾き v_1 = v̂(x, t) を測り、Euler で仮に進んだ点 x̃ = x - Δ · v_1 を作る
            velocity_1 = self._velocity(x, t_batch, memory, memory_pos)
            x_tilde = x - delta * velocity_1

            #   2c: 仮に進んだ点の傾き v_2 = v̂(x̃, t - Δ) を測る（x ではなく x̃、t ではなく t - Δ で測る）
            velocity_2 = self._velocity(x_tilde, next_t_batch, memory, memory_pos)

            #   2d: 2つの傾きの平均で、x から進み直す: x = x - Δ · (v_1 + v_2) / 2
            #       （x̃ から進むのではない。x̃ は v_2 を測るためだけに使う）
            velocity_average = (velocity_1 + velocity_2) / 2
            x = x - delta * velocity_average

        # Step 3: ループを抜けた x を返す（最後のステップの v_2 は t = 0 で測る）
        return x

    @torch.no_grad()
    def predict_chunk(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        # Step 1: 関節角を正規化し、self._encode_obs で memory を作る（ODE を解く前に1回だけ）
        state = obs["state"]  # (B, 14)
        state_norm = self.state_norm.normalize(state)  # (B, 14)
        image = obs["image"]  # (B, 3, H, W) uint8

        # shapeのLは、画像トークンの数 h*w + 1（関節角のトークン）で、k ではないことに注意
        memory, memory_pos = self._encode_obs(image, state_norm)  # (B, L, d_model), (1, L, d_model)

        # Step 2: 出発点 x_1 ~ N(0, I) を (B, k, ACTION_DIM) の乱数として生成する
        #         （torch.randn。device は memory に合わせる）
        batch_size = state.shape[0]
        x_1 = torch.randn((batch_size, self.chunk_size, ACTION_DIM), device=memory.device)

        # Step 3: self.solver が "euler" なら _sample_euler、"heun" なら _sample_heun
        #         （ステップ数はどちらも self.num_inference_steps）、それ以外は ValueError
        if self.solver == "euler":
            action_chunk_pred_norm = self._sample_euler(
                x_1, memory, memory_pos, self.num_inference_steps
            )
        elif self.solver == "heun":
            action_chunk_pred_norm = self._sample_heun(
                x_1, memory, memory_pos, self.num_inference_steps
            )
        else:
            raise ValueError(f"unknown solver: {self.solver} (expected one of {SOLVERS})")

        # Step 4: 行動の正規化を戻して（self.action_norm.denormalize）返す
        action_chunk_pred = self.action_norm.denormalize(action_chunk_pred_norm)  # (B, k, 14)

        return action_chunk_pred
