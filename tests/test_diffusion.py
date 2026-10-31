"""policies/diffusion.py のテスト（CPU・小さいモデル・小さい画像で動かす）。

uv run pytest tests/test_diffusion.py -v
uv run pytest tests/test_diffusion.py -v -k schedule   # 手順書の D2 の分だけ（q_sample / loss / ddpm / ddim も同様）

サンプラーのテストでは、デノイザを「正解のノイズを返す理想のデノイザ（oracle）」に差し替える。
データが1点 c しかない（x_0 が必ず c）なら、x_t と t から足されたノイズは
    ε = (x_t - √ᾱ_t · c) / √(1 - ᾱ_t)
と正確に求まる。このデノイザを使えば、正しいサンプラーは乱数によらず c にたどり着く。
学習済みのモデルがなくても、サンプラーの式だけを確かめられる。
"""

import math
from pathlib import Path
from typing import Any

import pytest
import torch

from pai_act.exps.checkpoint import build_policy, load_checkpoint, save_checkpoint
from pai_act.exps.config import EvalConfig, TrainConfig
from pai_act.policies.diffusion import DiffusionPolicy, cosine_beta_schedule

K = 5
T = 100
SMALL: dict[str, Any] = {
    "d_model": 32,
    "num_heads": 4,
    "dim_feedforward": 64,
    "num_encoder_layers": 1,
    "num_decoder_layers": 1,
    "dropout": 0.0,
}


# 理想のデノイザ（下の oracle_denoiser）は memory を使わないので、サンプラーには空のテンソルを渡す
NO_MEMORY = torch.empty(0)


def small_policy(**overrides) -> DiffusionPolicy:
    """CPU でもすぐ動く小さい DiffusionPolicy。正規化は恒等ではない値にしておく（正規化の漏れを見つけるため）。"""
    torch.manual_seed(0)
    policy = DiffusionPolicy(chunk_size=K, pretrained_backbone=False, **{**SMALL, **overrides})
    policy.state_norm.fit(torch.randn(100, 14) * 2 + 1)
    policy.action_norm.fit(torch.randn(100, 14) * 3 - 1)
    return policy.eval()


def fake_batch(b=2, h=64, w=96):
    torch.manual_seed(1)
    return {
        "image": torch.randint(0, 256, (b, 3, h, w), dtype=torch.uint8),
        "state": torch.randn(b, 14),
        "action": torch.randn(b, K, 14) * 3 - 1,
        "action_is_pad": torch.zeros(b, K, dtype=torch.bool),
    }


def oracle_denoiser(
    policy: DiffusionPolicy,
    x0_n: torch.Tensor,
    calls: list[int] | None = None,
    inputs: dict[int, torch.Tensor] | None = None,
):
    """x_0 が x0_n だと知っている理想のデノイザ。呼ばれた t を calls に、そのときの x_t を inputs に記録する。"""

    def denoise(x_t, t, memory, memory_pos):
        assert t.shape == (x_t.shape[0],), "t はサンプルごとに1つ (B,)"
        assert not t.is_floating_point(), "DDPM の t は整数（alphas_cumprod の添字に使う）"
        if calls is not None:
            calls.append(int(t[0]))
        if inputs is not None:
            inputs[int(t[0])] = x_t.clone()
        ab = policy.alphas_cumprod[t][:, None, None]
        return (x_t - ab.sqrt() * x0_n) / (1 - ab).sqrt()

    return denoise


# ---- D2: ノイズスケジュール ----


def test_schedule_cosine_formula():
    """cosine スケジュールが Nichol & Dhariwal 2021 の式（β の上限 0.999）と一致する。"""

    def f(u):
        return math.cos((u + 0.008) / 1.008 * math.pi / 2) ** 2

    expected = torch.tensor([min(1 - f((i + 1) / T) / f(i / T), 0.999) for i in range(T)])
    betas = cosine_beta_schedule(T)
    assert betas.shape == (T,)
    assert torch.allclose(betas.float(), expected.float(), atol=1e-6)


def test_schedule_buffers():
    """β_t・α_t・ᾱ_t がバッファとして揃っていて、ᾱ_t が 1 に近い値から 0 に近い値まで単調に減る。

    ᾱ_{T-1} ≈ 0 なので、最後の x_{T-1} はほぼ純粋なノイズになり、推論は N(0, I) から始められる。
    """
    policy = small_policy()
    buffers = dict(policy.named_buffers())
    for name in ("betas", "alphas", "alphas_cumprod"):
        assert name in buffers, f"{name} がバッファになっていない"
        assert buffers[name].shape == (T,)
    assert not any(n in dict(policy.named_parameters()) for n in ("betas", "alphas"))
    assert torch.allclose(policy.betas, cosine_beta_schedule(T).to(policy.betas.dtype))
    assert torch.allclose(policy.alphas, 1 - policy.betas)
    assert torch.allclose(policy.alphas_cumprod, torch.cumprod(policy.alphas, dim=0))
    ab = policy.alphas_cumprod
    assert (ab[1:] < ab[:-1]).all(), "ᾱ_t は単調に減る"
    assert ab[0] > 0.999 and ab[-1] < 1e-3


# ---- D3: forward process ----


def test_q_sample():
    """x_t = √ᾱ_t · x_0 + √(1 - ᾱ_t) · ε を、サンプルごとに違う t で計算できる。

    t = 0 ではほぼ x_0、t = T-1 ではほぼ ε になる。
    """
    policy = small_policy()
    torch.manual_seed(3)
    x0, noise = torch.randn(3, K, 14), torch.randn(3, K, 14)
    t = torch.tensor([0, 40, T - 1])
    x_t = policy.q_sample(x0, t, noise)
    assert x_t.shape == x0.shape
    for i in range(3):
        ab = policy.alphas_cumprod[t[i]]
        assert torch.allclose(x_t[i], ab.sqrt() * x0[i] + (1 - ab).sqrt() * noise[i], atol=1e-6)
    assert torch.allclose(x_t[0], x0[0], atol=0.15)  # √(1 - ᾱ_0) ≈ 0.025 なので少しだけずれる
    assert torch.allclose(x_t[2], noise[2], atol=0.05)


# ---- D4: 損失 ----


def test_loss_oracle_is_zero_and_mse(monkeypatch):
    """正解のノイズを返すデノイザなら損失は 0、全要素で 2 ずれたデノイザなら 4（L1 なら 2）。

    oracle は「正規化した行動」を x_0 として知っているので、compute_loss が行動を正規化し忘れたり、
    q_sample と違う式で x_t を作ったりすると 0 にならない。
    """
    policy = small_policy()
    batch = fake_batch()
    x0_n = policy.action_norm.normalize(batch["action"])
    calls: list[int] = []
    monkeypatch.setattr(policy, "_denoise", oracle_denoiser(policy, x0_n, calls))
    out = policy.compute_loss(batch)
    assert set(out) >= {"loss", "mse"}
    assert out["loss"].ndim == 0
    assert out["loss"].item() < 1e-6
    assert all(0 <= t < T for t in calls), "t は 0..T-1 から選ぶ"

    exact = oracle_denoiser(policy, x0_n)
    monkeypatch.setattr(policy, "_denoise", lambda *a: exact(*a) + 2.0)
    assert abs(policy.compute_loss(batch)["loss"].item() - 4.0) < 1e-4, "MSE になっていない"


def test_loss_ignores_padding(monkeypatch):
    """埋め草（action_is_pad = True）のステップでどれだけずれても、損失に入らない。"""
    policy = small_policy()
    batch = fake_batch()
    batch["action_is_pad"][:, -2:] = True
    exact = oracle_denoiser(policy, policy.action_norm.normalize(batch["action"]))

    def wrong_on_pad(*a):
        eps = exact(*a).clone()
        eps[:, -2:] += 100.0
        return eps

    monkeypatch.setattr(policy, "_denoise", wrong_on_pad)
    assert policy.compute_loss(batch)["loss"].item() < 1e-6


def test_loss_timesteps_vary(monkeypatch):
    """t はサンプルごとにランダムに選ぶ（バッチ全体で同じ t を使わない）。"""
    policy = small_policy()
    seen: list[torch.Tensor] = []
    real = policy._denoise

    def spy(x_t, t, memory, memory_pos):
        seen.append(t.clone())
        return real(x_t, t, memory, memory_pos)

    monkeypatch.setattr(policy, "_denoise", spy)
    policy.compute_loss(fake_batch(b=16))
    assert len(seen) == 1, "1回の compute_loss でデノイザは1回だけ呼ぶ"
    assert seen[0].unique().numel() > 1


def test_loss_grads_reach_all_params():
    """compute_loss から全パラメータに勾配が流れる。"""
    policy = small_policy().train()
    out = policy.compute_loss(fake_batch())
    out["loss"].backward()
    for name, p in policy.named_parameters():
        assert p.grad is not None, f"{name} に勾配が流れていない（使われていない層がある？）"


# ---- D5: DDPM サンプラー ----


def test_ddpm_oracle_recovers_x0(monkeypatch):
    """理想のデノイザを使うと、DDPM は t = T-1, ..., 0 の T 回で x_0 にたどり着く。

    途中で足す乱数の大きさ（分散）や、最後のステップで乱数を足さないことを間違えると、c に戻らない。
    """
    policy = small_policy(sampler="ddpm")
    torch.manual_seed(4)
    c_n = torch.randn(2, K, 14)
    calls: list[int] = []
    monkeypatch.setattr(policy, "_denoise", oracle_denoiser(policy, c_n, calls))
    x = policy._sample_ddpm(torch.randn(2, K, 14), NO_MEMORY, NO_MEMORY)
    assert calls == list(range(T - 1, -1, -1))
    assert torch.allclose(x, c_n, atol=1e-3)


def test_ddpm_intermediate_distribution(monkeypatch):
    """理想のデノイザを使うと、DDPM の途中の x_t は q(x_t | x_0 = c) = N(√ᾱ_t · c, (1 - ᾱ_t) I) に従う。

    最後のステップは途中の状態によらず c に戻るので、上のテストだけでは途中で足す乱数の誤りが見つからない。
    そこで途中の x_t を標準化し、平均 0・標準偏差 1 になっているかを多数のサンプルで確かめる。
    分散を β̃_t ではなく β_t にした場合（t = 1 で標準偏差 1.3 程度）、標準偏差の代わりに分散を掛けた場合、
    乱数を足し忘れた場合は、ここで見つかる。
    """
    policy = small_policy()
    torch.manual_seed(7)
    c_n = torch.randn(400, K, 14)
    inputs: dict[int, torch.Tensor] = {}
    monkeypatch.setattr(policy, "_denoise", oracle_denoiser(policy, c_n, inputs=inputs))
    policy._sample_ddpm(torch.randn(400, K, 14), NO_MEMORY, NO_MEMORY)
    for t in (50, 20, 5, 1):
        ab = policy.alphas_cumprod[t]
        z = (inputs[t] - ab.sqrt() * c_n) / (1 - ab).sqrt()
        assert abs(z.mean().item()) < 0.05, f"t={t}: 平均 {z.mean().item():.3f}"
        assert abs(z.std().item() - 1) < 0.05, f"t={t}: 標準偏差 {z.std().item():.3f}"


def test_ddpm_predict_chunk_units(monkeypatch):
    """predict_chunk は、サンプラーが出した正規化済みの行動を、シミュレータの単位に戻して返す。"""
    policy = small_policy(sampler="ddpm")
    torch.manual_seed(5)
    c_n = torch.randn(2, K, 14)
    monkeypatch.setattr(policy, "_denoise", oracle_denoiser(policy, c_n))
    batch = fake_batch()
    out = policy.predict_chunk(batch)
    assert out.shape == (2, K, 14)
    assert torch.allclose(out, policy.action_norm.denormalize(c_n), atol=1e-2)


def test_ddpm_is_stochastic():
    """DDPM は途中で乱数を足すので、出発点 x_T が同じでも、乱数が違えば結果が変わる。"""
    policy = small_policy()
    memory, memory_pos = policy._encode_obs(fake_batch()["image"], torch.randn(2, 14))
    x_T = torch.randn(2, K, 14)
    torch.manual_seed(10)
    a = policy._sample_ddpm(x_T.clone(), memory, memory_pos)
    torch.manual_seed(11)
    b = policy._sample_ddpm(x_T.clone(), memory, memory_pos)
    assert not torch.allclose(a, b)


# ---- D6: DDIM サンプラー ----


@pytest.mark.parametrize(
    ("num_steps", "expected_ts"),
    [(10, list(range(99, -1, -10))), (5, [99, 79, 59, 39, 19]), (100, list(range(99, -1, -1)))],
)
def test_ddim_oracle_recovers_x0(monkeypatch, num_steps, expected_ts):
    """理想のデノイザを使うと、DDIM は S 回（t = T-1 から T/S 刻み）で x_0 にたどり着く。"""
    policy = small_policy()
    torch.manual_seed(6)
    c_n = torch.randn(2, K, 14)
    calls: list[int] = []
    monkeypatch.setattr(policy, "_denoise", oracle_denoiser(policy, c_n, calls))
    x = policy._sample_ddim(torch.randn(2, K, 14), NO_MEMORY, NO_MEMORY, num_steps)
    assert calls == expected_ts
    assert torch.allclose(x, c_n, atol=1e-3)


def test_ddim_intermediate_keeps_noise(monkeypatch):
    """理想のデノイザを使うと、η = 0 の DDIM では途中の全ステップで x_t = √ᾱ_t · c + √(1 - ᾱ_t) · ε_T になる。

    ε_T は出発点 x_T に入っていたノイズで、DDIM は「同じノイズのまま ᾱ だけを変えていく」更新をする。
    途中のステップで x_0 の推定や係数を誤ると、どこかのステップでこの関係が崩れる。
    """
    policy = small_policy()
    torch.manual_seed(8)
    c_n = torch.randn(2, K, 14)
    inputs: dict[int, torch.Tensor] = {}
    monkeypatch.setattr(policy, "_denoise", oracle_denoiser(policy, c_n, inputs=inputs))
    x_T = torch.randn(2, K, 14)
    policy._sample_ddim(x_T.clone(), NO_MEMORY, NO_MEMORY, 10)
    ab_T = policy.alphas_cumprod[T - 1]
    eps_T = (x_T - ab_T.sqrt() * c_n) / (1 - ab_T).sqrt()
    for t, x_t in inputs.items():
        ab = policy.alphas_cumprod[t]
        assert torch.allclose(x_t, ab.sqrt() * c_n + (1 - ab).sqrt() * eps_T, atol=1e-3), f"t={t}"


def test_ddim_is_deterministic():
    """η = 0 の DDIM は乱数を足さないので、出発点 x_T が同じなら、乱数の状態によらず同じ結果になる。"""
    policy = small_policy()
    memory, memory_pos = policy._encode_obs(fake_batch()["image"], torch.randn(2, 14))
    x_T = torch.randn(2, K, 14)
    torch.manual_seed(10)
    a = policy._sample_ddim(x_T.clone(), memory, memory_pos, 10)
    torch.manual_seed(11)
    b = policy._sample_ddim(x_T.clone(), memory, memory_pos, 10)
    assert torch.allclose(a, b)


def test_ddim_predict_chunk_sampler_switch(monkeypatch):
    """predict_chunk は self.sampler で DDPM / DDIM を切り替え、知らない名前ならエラーにする。"""
    policy = small_policy()
    batch = fake_batch()
    calls: list[int] = []
    c_n = torch.zeros(2, K, 14)
    monkeypatch.setattr(policy, "_denoise", oracle_denoiser(policy, c_n, calls))
    for sampler, steps, n_calls in (("ddim", 10, 10), ("ddim", 20, 20), ("ddpm", 10, T)):
        calls.clear()
        policy.sampler, policy.num_inference_steps = sampler, steps
        assert policy.predict_chunk(batch).shape == (2, K, 14)
        assert len(calls) == n_calls, f"{sampler}（S={steps}）でデノイザを呼んだ回数"
    policy.sampler = "unknown"
    with pytest.raises(ValueError):
        policy.predict_chunk(batch)


# ---- D7b: x̂_0 のクリップ ----


def zero_denoiser(inputs: list[torch.Tensor] | None = None):
    """常に ε̂ = 0 を返す、でたらめなデノイザ。呼ばれたときの x_t を inputs に記録する。

    最初のステップ（t = T-1）で x̂_0 = x_T / √ᾱ_{T-1} ≈ 2000 · x_T となり、クリップしなければ出力が発散する。
    乱数を使わないので、サンプラーの乱数の使い方に影響しない。
    """

    def denoise(x_t, t, memory, memory_pos):
        if inputs is not None:
            inputs.append(x_t.clone())
        return torch.zeros_like(x_t)

    return denoise


def run_sampler(policy: DiffusionPolicy, sampler: str, x_T: torch.Tensor) -> torch.Tensor:
    if sampler == "ddpm":
        return policy._sample_ddpm(x_T, NO_MEMORY, NO_MEMORY)
    return policy._sample_ddim(x_T, NO_MEMORY, NO_MEMORY, 10)


@pytest.mark.parametrize("sampler", ["ddpm", "ddim"])
def test_clip_bounds_every_step(monkeypatch, sampler):
    """デノイザがでたらめでも、x̂_0 を毎ステップクリップすれば、途中の x_t も出力も発散しない。

    出力は最後のステップの x̂_0（をクリップしたもの）なので [-x0_clip, x0_clip] に収まる。
    最後に1回だけクリップする実装では、途中の x_t が数百〜数千になるので、ここで見つかる。
    x0_clip = 0 でクリップしないと発散することも確かめる。
    """
    torch.manual_seed(12)
    x_T = torch.randn(2, K, 14)

    policy = small_policy(x0_clip=3.0)
    inputs: list[torch.Tensor] = []
    monkeypatch.setattr(policy, "_denoise", zero_denoiser(inputs))
    torch.manual_seed(13)
    out = run_sampler(policy, sampler, x_T.clone())
    assert out.abs().max() <= 3.0 * (1 + 1e-3), "出力が [-x0_clip, x0_clip] に収まっていない"
    for i, x_t in enumerate(inputs[1:], start=1):
        assert x_t.abs().max() < 10.0, f"{i + 1} 回目のデノイザの入力 x_t が発散している"

    policy.x0_clip = 0.0
    torch.manual_seed(13)
    out = run_sampler(policy, sampler, x_T.clone())
    assert out.abs().max() > 100.0, "x0_clip = 0 ではクリップしない（ε̂ の誤差が約 2000 倍される）"


def test_clip_off_ddpm_matches_eps_form(monkeypatch):
    """クリップしなければ、x̂_0 を使う形の DDPM の平均は、ε̂ を直接使う式と同じ値になる。

    float64 で、乱数を使わない（x_t から決まる）デノイザを使い、同じ乱数の状態から両方の式で辿って比べる。
    x̂_0 の係数・x_t の係数・t = 0 での ᾱ_{-1} = 1 の扱いを誤ると一致しない。
    """
    policy = small_policy(x0_clip=0.0).double()
    # バッファは float32 で作った値なので、.double() だけでは ᾱ_t = α_t · ᾱ_{t-1} が 5e-8 の精度でしか
    # 成り立たない。2つの式が一致するのはこの恒等式のおかげで、ずれは 1/(1 - ᾱ_t) 倍される
    # （t = 0 で約 1600 倍）。そのため最後のほうのステップで相対 1e-5 ほどずれて比べられない。float64 で作り直す
    betas = cosine_beta_schedule(T).double()
    policy.betas.copy_(betas)
    policy.alphas.copy_(1 - betas)
    policy.alphas_cumprod.copy_(torch.cumprod(1 - betas, dim=0))
    monkeypatch.setattr(policy, "_denoise", lambda x_t, t, m, mp: 0.9 * torch.tanh(x_t))
    torch.manual_seed(14)
    x_T = torch.randn(2, K, 14, dtype=torch.float64)

    torch.manual_seed(15)
    out = policy._sample_ddpm(x_T.clone(), NO_MEMORY, NO_MEMORY)

    torch.manual_seed(15)
    x = x_T.clone()
    b, a, ab = policy.betas, policy.alphas, policy.alphas_cumprod
    for t in reversed(range(T)):
        eps = 0.9 * torch.tanh(x)
        x = (x - b[t] / (1 - ab[t]).sqrt() * eps) / a[t].sqrt()
        if t > 0:
            x = x + ((1 - ab[t - 1]) / (1 - ab[t]) * b[t]).sqrt() * torch.randn_like(x)
    assert torch.allclose(out, x, rtol=1e-6, atol=1e-8)


# ---- D11: DDIM に乱数を足す（η > 0） ----


def test_ddim_eta1_full_steps_matches_ddpm(monkeypatch):
    """η = 1 かつ S = T の DDIM は、DDPM と同じ更新になる（σ² = β̃_t になり、平均も一致する）。

    乱数の状態を揃えて両方を回し、途中の全ステップの x_t が一致するかを見る。
    理想のデノイザでは x̂_0 = c なので、クリップは効かない（c は ±6 に収まる）。
    σ の式の誤り（η を掛け忘れる・2乗と平方根を取り違える・ᾱ_t / ᾱ_{t'} を逆にする）や、
    ε̂ の係数を √(1 - ᾱ_{t'}) のままにした誤りは、ここで見つかる。
    """
    policy = small_policy(ddim_eta=1.0)
    torch.manual_seed(12)
    c_n = torch.randn(2, K, 14)
    x_T = torch.randn(2, K, 14)
    ddpm_inputs: dict[int, torch.Tensor] = {}
    ddim_inputs: dict[int, torch.Tensor] = {}
    monkeypatch.setattr(policy, "_denoise", oracle_denoiser(policy, c_n, inputs=ddpm_inputs))
    torch.manual_seed(13)
    policy._sample_ddpm(x_T.clone(), NO_MEMORY, NO_MEMORY)
    monkeypatch.setattr(policy, "_denoise", oracle_denoiser(policy, c_n, inputs=ddim_inputs))
    torch.manual_seed(13)
    policy._sample_ddim(x_T.clone(), NO_MEMORY, NO_MEMORY, T)
    assert sorted(ddim_inputs) == sorted(ddpm_inputs)
    for t, x_t in ddpm_inputs.items():
        assert torch.allclose(ddim_inputs[t], x_t, atol=1e-4), f"t={t}"


@pytest.mark.parametrize("eta", [0.5, 1.0])
def test_ddim_eta_intermediate_distribution(monkeypatch, eta):
    """理想のデノイザを使うと、η によらず途中の x_t は N(√ᾱ_t · c, (1 - ᾱ_t) I) に従い、
    隣り合うステップの（標準化した）ノイズの相関は √(1 - ᾱ_{t'} - σ²) / √(1 - ᾱ_{t'}) になる。

    η = 0 では相関が 1（同じノイズのまま。test_ddim_intermediate_keeps_noise）、η = 1 の最初のステップでは
    約 0.003（ほぼ全部が新しい乱数に入れ替わる）。S < T でも σ の式が正しいかを、この相関で確かめる。
    ε̂ の係数を小さくし忘れると標準偏差が 1 を超え、乱数を足し忘れると相関が 1 になる。
    """
    policy = small_policy(ddim_eta=eta)
    torch.manual_seed(14)
    c_n = torch.randn(400, K, 14)
    inputs: dict[int, torch.Tensor] = {}
    monkeypatch.setattr(policy, "_denoise", oracle_denoiser(policy, c_n, inputs=inputs))
    policy._sample_ddim(torch.randn(400, K, 14), NO_MEMORY, NO_MEMORY, 10)
    ab = policy.alphas_cumprod

    def noise(t: int) -> torch.Tensor:
        return (inputs[t] - ab[t].sqrt() * c_n) / (1 - ab[t]).sqrt()

    for t in range(T - 1, 9, -10):  # (t, t') = (99, 89), ..., (19, 9)
        t_prime = t - 10
        z, z_prime = noise(t), noise(t_prime)
        assert abs(z_prime.mean().item()) < 0.05, f"t'={t_prime}: 平均 {z_prime.mean().item():.3f}"
        assert abs(z_prime.std().item() - 1) < 0.05, f"t'={t_prime}: 標準偏差 {z_prime.std():.3f}"
        sigma2 = eta**2 * (1 - ab[t_prime]) / (1 - ab[t]) * (1 - ab[t] / ab[t_prime])
        expected = ((1 - ab[t_prime] - sigma2) / (1 - ab[t_prime])).sqrt().item()
        corr = ((z - z.mean()) * (z_prime - z_prime.mean())).mean() / (z.std() * z_prime.std())
        assert abs(corr.item() - expected) < 0.03, (
            f"t={t} → t'={t_prime}: 相関 {corr.item():.3f}（期待値 {expected:.3f}）"
        )


def test_ddim_eta_recovers_x0(monkeypatch):
    """η > 0 でも、理想のデノイザなら最後のステップ（ᾱ_{t'} = 1、σ = 0）で x_0 にたどり着く。"""
    policy = small_policy(ddim_eta=1.0)
    torch.manual_seed(15)
    c_n = torch.randn(2, K, 14)
    calls: list[int] = []
    monkeypatch.setattr(policy, "_denoise", oracle_denoiser(policy, c_n, calls))
    x = policy._sample_ddim(torch.randn(2, K, 14), NO_MEMORY, NO_MEMORY, 10)
    assert calls == list(range(99, -1, -10))
    assert torch.allclose(x, c_n, atol=1e-3)


def test_ddim_eta_is_stochastic():
    """η > 0 の DDIM は途中で乱数を足すので、出発点 x_T が同じでも、乱数が違えば結果が変わる。"""
    policy = small_policy(ddim_eta=0.5)
    memory, memory_pos = policy._encode_obs(fake_batch()["image"], torch.randn(2, 14))
    x_T = torch.randn(2, K, 14)
    torch.manual_seed(10)
    a = policy._sample_ddim(x_T.clone(), memory, memory_pos, 10)
    torch.manual_seed(11)
    b = policy._sample_ddim(x_T.clone(), memory, memory_pos, 10)
    assert not torch.allclose(a, b)


# ---- 配線（config・build_policy・チェックポイント・評価の上書き） ----


def test_build_policy_and_checkpoint_roundtrip(tmp_path):
    """設定から DiffusionPolicy を作れ、保存して読み直したモデルが同じ予測を返す。"""
    cfg = TrainConfig(policy="diffusion", chunk_size=K, num_inference_steps=5, x0_clip=4.0, **SMALL)
    policy = build_policy(cfg, pretrained_backbone=False).eval()
    assert isinstance(policy, DiffusionPolicy)
    assert (policy.sampler, policy.num_inference_steps, policy.x0_clip) == ("ddim", 5, 4.0)
    policy.action_norm.fit(torch.randn(100, 14) * 3 + 1)
    save_checkpoint(tmp_path / "ckpt.pt", policy, cfg, step=1)
    restored, _ = load_checkpoint(tmp_path / "ckpt.pt")
    batch = fake_batch()
    torch.manual_seed(0)  # 初期ノイズ x_T を揃える
    a = policy.predict_chunk(batch)
    torch.manual_seed(0)
    b = restored.predict_chunk(batch)
    assert torch.allclose(a, b, atol=1e-5)


def test_build_policy_ddim_eta():
    """設定の ddim_eta がポリシーに渡る。既定は 0（η を足す前の DDIM と同じ）。"""
    cfg = TrainConfig(policy="diffusion", chunk_size=K, ddim_eta=0.5, **SMALL)
    assert build_policy(cfg, pretrained_backbone=False).ddim_eta == 0.5
    assert TrainConfig.ddim_eta == 0.0


def test_eval_sampler_overrides():
    """評価の --sampler / --num-inference-steps で、学習し直さずにサンプラーを切り替えられる。"""
    from pai_act.exps.evaluate import apply_sampler_overrides
    from pai_act.policies.baseline import BaselinePolicy

    policy = small_policy()
    apply_sampler_overrides(policy, EvalConfig(ckpt=Path("x"), sampler="ddpm"))
    assert (policy.sampler, policy.num_inference_steps) == ("ddpm", 10)  # 指定しない方はそのまま
    apply_sampler_overrides(policy, EvalConfig(ckpt=Path("x"), num_inference_steps=20))
    assert (policy.sampler, policy.num_inference_steps) == ("ddpm", 20)
    assert policy.x0_clip == 6.0
    apply_sampler_overrides(policy, EvalConfig(ckpt=Path("x"), x0_clip=0.0))
    assert policy.x0_clip == 0.0  # 0 も「指定した」扱い（None だけが省略）
    assert policy.ddim_eta == 0.0
    apply_sampler_overrides(policy, EvalConfig(ckpt=Path("x"), ddim_eta=1.0))
    assert policy.ddim_eta == 1.0
    with pytest.raises(ValueError):
        apply_sampler_overrides(
            BaselinePolicy(chunk_size=K, pretrained_backbone=False),
            EvalConfig(ckpt=Path("x"), sampler="ddim"),
        )
