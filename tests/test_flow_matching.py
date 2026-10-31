"""policies/flow_matching.py のテスト（CPU・小さいモデル・小さい画像で動かす）。

uv run pytest tests/test_flow_matching.py -v
uv run pytest tests/test_flow_matching.py -v -k interpolate   # 手順書の F1 の分だけ（loss / euler / heun も同様）

ソルバーのテストでは、ネットワークの速度 _velocity を、式で書ける速度に差し替える。
    理想の速度（oracle）: データが1点 c しかない（x_0 が必ず c）なら、x_t = (1 - t) · c + t · ε から
        u = ε - c = (x_t - c) / t
      と x_t と t だけで正確に求まる。出発点 x_1 から c までの直線の上では、この速度はずっと同じ値になる。
    線形の ODE: v(x, t) = x とすると、dx/dt = x の解は x(t) = x(1) · e^{t-1}。各ソルバーの1ステップは
      x に定数を掛ける操作になり、N ステップ後の値を式で書ける（Euler: (1 - Δ)^N 倍、Heun: (1 - Δ + Δ²/2)^N 倍）。
学習済みのモデルがなくても、ソルバーの式だけを確かめられる。
"""

import math
from pathlib import Path
from typing import Any

import pytest
import torch

from pai_act.exps.checkpoint import build_policy, load_checkpoint, save_checkpoint
from pai_act.exps.config import EvalConfig, TrainConfig
from pai_act.policies.flow_matching import FlowMatchingPolicy

K = 5
SMALL: dict[str, Any] = {
    "d_model": 32,
    "num_heads": 4,
    "dim_feedforward": 64,
    "num_encoder_layers": 1,
    "num_decoder_layers": 1,
    "dropout": 0.0,
}

# 差し替えた速度は memory を使わないので、ソルバーには空のテンソルを渡す
NO_MEMORY = torch.empty(0)


def small_policy(**overrides) -> FlowMatchingPolicy:
    """CPU でもすぐ動く小さい FlowMatchingPolicy。正規化は恒等ではない値にしておく（正規化の漏れを見つけるため）。"""
    torch.manual_seed(0)
    policy = FlowMatchingPolicy(chunk_size=K, pretrained_backbone=False, **{**SMALL, **overrides})
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


def check_t(x_t: torch.Tensor, t: torch.Tensor) -> None:
    assert t.shape == (x_t.shape[0],), "t はサンプルごとに1つ (B,)"
    assert t.is_floating_point(), "Flow Matching の t は 0〜1 の実数"


def oracle_velocity(
    x0_n: torch.Tensor,
    calls: list[float] | None = None,
    inputs: list[tuple[float, torch.Tensor]] | None = None,
):
    """x_0 が x0_n だと知っている理想の速度 (x_t - x0_n) / t。

    呼ばれた t を calls に、(t, x_t) を inputs に記録する。t = 0 では割り算ができないので使えない。
    """

    def velocity(x_t, t, memory, memory_pos):
        check_t(x_t, t)
        if calls is not None:
            calls.append(float(t[0]))
        if inputs is not None:
            inputs.append((float(t[0]), x_t.clone()))
        return (x_t - x0_n) / t[:, None, None]

    return velocity


def linear_velocity(calls: list[float] | None = None):
    """dx/dt = x の速度。呼ばれた t を calls に記録する。"""

    def velocity(x_t, t, memory, memory_pos):
        check_t(x_t, t)
        if calls is not None:
            calls.append(float(t[0]))
        return x_t.clone()

    return velocity


# ---- F1: 直線の上の点と、時刻の拡大 ----


def test_interpolate():
    """x_t = (1 - t) · x_0 + t · ε を、サンプルごとに違う t で計算できる。t = 0 で x_0、t = 1 で ε になる。"""
    policy = small_policy()
    torch.manual_seed(3)
    x0, noise = torch.randn(3, K, 14), torch.randn(3, K, 14)
    t = torch.tensor([0.0, 0.3, 1.0])
    x_t = policy.interpolate(x0, t, noise)
    assert x_t.shape == x0.shape
    for i in range(3):
        assert torch.allclose(x_t[i], (1 - t[i]) * x0[i] + t[i] * noise[i], atol=1e-6)
    assert torch.allclose(x_t[0], x0[0])
    assert torch.allclose(x_t[2], noise[2])


def test_interpolate_velocity_is_constant():
    """直線なので、x_t を t で微分した速度は、どの t でも ε - x_0 になる（差分で確かめる）。"""
    policy = small_policy()
    torch.manual_seed(4)
    x0, noise = torch.randn(1, K, 14), torch.randn(1, K, 14)
    h = 1e-2
    for t in (0.1, 0.5, 0.9):
        a = policy.interpolate(x0, torch.tensor([t]), noise)
        b = policy.interpolate(x0, torch.tensor([t + h]), noise)
        assert torch.allclose((b - a) / h, noise - x0, atol=1e-3), f"t={t}"


def test_velocity_scales_time(monkeypatch):
    """_velocity は t に time_scale を掛けてデノイザに渡す。整数に丸めない。"""
    policy = small_policy()
    seen: list[torch.Tensor] = []

    def spy(x_t, t, memory, memory_pos):
        seen.append(t.clone())
        return x_t * 2

    monkeypatch.setattr(policy, "_denoise", spy)
    x = torch.randn(3, K, 14)
    out = policy._velocity(x, torch.tensor([0.0, 0.255, 1.0]), NO_MEMORY, NO_MEMORY)
    assert torch.allclose(out, x * 2), "デノイザの出力をそのまま返す"
    assert len(seen) == 1
    assert torch.allclose(seen[0], torch.tensor([0.0, 25.5, 100.0])), "t × time_scale（既定 100）"


# ---- F2: 損失 ----


def test_loss_oracle_is_zero_and_mse(monkeypatch):
    """正解の速度を返す _velocity なら損失は 0、全要素で 2 ずれた _velocity なら 4（L1 なら 2）。

    理想の速度は「正規化した行動」を x_0 として知っているので、compute_loss が行動を正規化し忘れたり、
    interpolate と違う式で x_t を作ったり、速度の向き（ε - x_0 と x_0 - ε）を逆にしたりすると 0 にならない。
    """
    policy = small_policy()
    batch = fake_batch()
    x0_n = policy.action_norm.normalize(batch["action"])
    calls: list[float] = []
    monkeypatch.setattr(policy, "_velocity", oracle_velocity(x0_n, calls))
    out = policy.compute_loss(batch)
    assert set(out) >= {"loss", "mse"}
    assert out["loss"].ndim == 0
    assert out["loss"].item() < 1e-6
    assert all(0 <= t <= 1 for t in calls), (
        "_velocity には 0〜1 の t を渡す（拡大は _velocity の中）"
    )

    exact = oracle_velocity(x0_n)
    monkeypatch.setattr(policy, "_velocity", lambda *a: exact(*a) + 2.0)
    assert abs(policy.compute_loss(batch)["loss"].item() - 4.0) < 1e-4, "MSE になっていない"


def test_loss_ignores_padding(monkeypatch):
    """埋め草（action_is_pad = True）のステップでどれだけずれても、損失に入らない。"""
    policy = small_policy()
    batch = fake_batch()
    batch["action_is_pad"][:, -2:] = True
    exact = oracle_velocity(policy.action_norm.normalize(batch["action"]))

    def wrong_on_pad(*a):
        v = exact(*a).clone()
        v[:, -2:] += 100.0
        return v

    monkeypatch.setattr(policy, "_velocity", wrong_on_pad)
    assert policy.compute_loss(batch)["loss"].item() < 1e-6


def test_loss_times_vary(monkeypatch):
    """t はサンプルごとに [0, 1) からランダムに選ぶ（バッチ全体で同じ t を使わない。整数にしない）。"""
    policy = small_policy()
    seen: list[torch.Tensor] = []
    real = policy._velocity

    def spy(x_t, t, memory, memory_pos):
        check_t(x_t, t)
        seen.append(t.clone())
        return real(x_t, t, memory, memory_pos)

    monkeypatch.setattr(policy, "_velocity", spy)
    policy.compute_loss(fake_batch(b=16))
    assert len(seen) == 1, "1回の compute_loss でデノイザは1回だけ呼ぶ"
    t = seen[0]
    assert t.unique().numel() > 1
    assert ((t >= 0) & (t <= 1)).all()
    assert (t != t.round()).any(), "t が 0 か 1 に丸められている"


def test_loss_grads_reach_all_params():
    """compute_loss から全パラメータに勾配が流れる。"""
    policy = small_policy().train()
    out = policy.compute_loss(fake_batch())
    out["loss"].backward()
    for name, p in policy.named_parameters():
        assert p.grad is not None, f"{name} に勾配が流れていない（使われていない層がある？）"


# ---- F3: Euler 法 ----


@pytest.mark.parametrize("num_steps", [1, 4, 10])
def test_euler_oracle_recovers_x0(monkeypatch, num_steps):
    """理想の速度を使うと、Euler は t = 1, 1 - Δ, ..., Δ の N 回で x_0 にたどり着く。

    データが1点なら、x_1 から c への道筋は直線で、速度はずっと同じ値になる。
    傾きが変わらないので、Euler は N = 1 でも誤差なく c に着く。
    """
    policy = small_policy()
    torch.manual_seed(6)
    c_n = torch.randn(2, K, 14)
    calls: list[float] = []
    monkeypatch.setattr(policy, "_velocity", oracle_velocity(c_n, calls))
    x = policy._sample_euler(torch.randn(2, K, 14), NO_MEMORY, NO_MEMORY, num_steps)
    assert calls == pytest.approx([1 - i / num_steps for i in range(num_steps)], abs=1e-6)
    assert torch.allclose(x, c_n, atol=1e-4)


def test_euler_intermediate_on_line(monkeypatch):
    """理想の速度を使うと、途中の全ステップで x_t = (1 - t) · c + t · x_1 になる（c と出発点 x_1 を結ぶ直線の上）。

    刻み幅や向き（引くか足すか）を誤ると、どこかのステップでこの関係が崩れる。
    """
    policy = small_policy()
    torch.manual_seed(8)
    c_n = torch.randn(2, K, 14)
    inputs: list[tuple[float, torch.Tensor]] = []
    monkeypatch.setattr(policy, "_velocity", oracle_velocity(c_n, inputs=inputs))
    x_1 = torch.randn(2, K, 14)
    policy._sample_euler(x_1.clone(), NO_MEMORY, NO_MEMORY, 10)
    assert len(inputs) == 10
    for t, x_t in inputs:
        assert torch.allclose(x_t, (1 - t) * c_n + t * x_1, atol=1e-4), f"t={t:.2f}"


def test_euler_linear_ode(monkeypatch):
    """dx/dt = x を Euler で解くと、1ステップごとに x が (1 - Δ) 倍になる。N = 4 なら 0.75⁴ ≈ 0.316 倍。

    正しい解は e^{-1} ≈ 0.368 倍で、Euler は傾きをステップの始めでしか測らないので少しずれる。
    """
    policy = small_policy()
    monkeypatch.setattr(policy, "_velocity", linear_velocity())
    x_1 = torch.randn(2, K, 14)
    x = policy._sample_euler(x_1.clone(), NO_MEMORY, NO_MEMORY, 4)
    assert torch.allclose(x, x_1 * 0.75**4, atol=1e-5)


def test_euler_is_deterministic():
    """ODE のソルバーは乱数を足さないので、出発点 x_1 が同じなら、乱数の状態によらず同じ結果になる。"""
    policy = small_policy()
    memory, memory_pos = policy._encode_obs(fake_batch()["image"], torch.randn(2, 14))
    x_1 = torch.randn(2, K, 14)
    torch.manual_seed(10)
    a = policy._sample_euler(x_1.clone(), memory, memory_pos, 10)
    torch.manual_seed(11)
    b = policy._sample_euler(x_1.clone(), memory, memory_pos, 10)
    assert torch.allclose(a, b)


def test_euler_predict_chunk_units(monkeypatch):
    """predict_chunk は、ソルバーが出した正規化済みの行動を、シミュレータの単位に戻して返す。"""
    policy = small_policy(solver="euler")
    torch.manual_seed(5)
    c_n = torch.randn(2, K, 14)
    monkeypatch.setattr(policy, "_velocity", oracle_velocity(c_n))
    out = policy.predict_chunk(fake_batch())
    assert out.shape == (2, K, 14)
    assert torch.allclose(out, policy.action_norm.denormalize(c_n), atol=1e-3)


# ---- F4: Heun 法 ----


def test_heun_times(monkeypatch):
    """Heun は1ステップでデノイザを2回呼ぶ: いまの時刻 t と、次の時刻 t - Δ。最後の2回目は t = 0。"""
    policy = small_policy()
    calls: list[float] = []
    monkeypatch.setattr(policy, "_velocity", linear_velocity(calls))
    policy._sample_heun(torch.randn(2, K, 14), NO_MEMORY, NO_MEMORY, 4)
    assert calls == pytest.approx([1.0, 0.75, 0.75, 0.5, 0.5, 0.25, 0.25, 0.0], abs=1e-6)


def test_heun_linear_ode(monkeypatch):
    """dx/dt = x を Heun で解くと、1ステップごとに x が (1 - Δ + Δ²/2) 倍になる。

    1ステップの中身（Δ = 0.25）:
        v_1 = x                     → 仮に進んだ点 x̃ = x - Δ · x = 0.75 · x
        v_2 = x̃ = 0.75 · x
        x - Δ · (v_1 + v_2) / 2 = x - 0.25 · 0.875 · x = 0.78125 · x     （= 1 - Δ + Δ²/2）
    N = 4 なら 0.78125⁴ ≈ 0.373 倍で、正しい解 e^{-1} ≈ 0.368 に Euler（0.316）より近い。
    v_2 を x̃ ではなく x で測ったり、x̃ から進み直したりすると、この値にならない。
    """
    policy = small_policy()
    monkeypatch.setattr(policy, "_velocity", linear_velocity())
    x_1 = torch.randn(2, K, 14)
    x = policy._sample_heun(x_1.clone(), NO_MEMORY, NO_MEMORY, 4)
    assert torch.allclose(x, x_1 * 0.78125**4, atol=1e-5)
    euler_err = abs(0.75**4 - math.exp(-1))
    heun_err = abs(0.78125**4 - math.exp(-1))
    assert heun_err < euler_err / 5  # 2次の方法なので誤差が1桁近く小さい（0.052 と 0.005）


def test_heun_is_deterministic():
    """Heun も乱数を足さない。"""
    policy = small_policy()
    memory, memory_pos = policy._encode_obs(fake_batch()["image"], torch.randn(2, 14))
    x_1 = torch.randn(2, K, 14)
    torch.manual_seed(10)
    a = policy._sample_heun(x_1.clone(), memory, memory_pos, 5)
    torch.manual_seed(11)
    b = policy._sample_heun(x_1.clone(), memory, memory_pos, 5)
    assert torch.allclose(a, b)


@pytest.mark.parametrize("solver", ["euler", "heun"])
@pytest.mark.parametrize("num_steps", [0, -2])
def test_solver_rejects_nonpositive_steps(monkeypatch, solver, num_steps):
    """ステップ数が 1 未満ならエラーにする。

    0 では 1 / 0 の割り算で落ちるが、負の値ではループが1回も回らず、出発点のノイズがエラーなしで返ってしまう。
    評価の --num-inference-steps を打ち間違えたときに、ノイズを行動として実行しないようにする。
    """
    policy = small_policy()
    monkeypatch.setattr(policy, "_velocity", linear_velocity())
    sample = policy._sample_euler if solver == "euler" else policy._sample_heun
    with pytest.raises(AssertionError):
        sample(torch.randn(2, K, 14), NO_MEMORY, NO_MEMORY, num_steps)


def test_predict_chunk_solver_switch(monkeypatch):
    """predict_chunk は self.solver で Euler / Heun を切り替え、知らない名前ならエラーにする。"""
    policy = small_policy()
    batch = fake_batch()
    calls: list[float] = []
    monkeypatch.setattr(policy, "_velocity", linear_velocity(calls))
    for solver, steps, n_calls in (("euler", 10, 10), ("euler", 3, 3), ("heun", 5, 10)):
        calls.clear()
        policy.solver, policy.num_inference_steps = solver, steps
        assert policy.predict_chunk(batch).shape == (2, K, 14)
        assert len(calls) == n_calls, f"{solver}（N={steps}）でデノイザを呼んだ回数"
    policy.solver = "unknown"
    with pytest.raises(ValueError):
        policy.predict_chunk(batch)


# ---- 配線（config・build_policy・チェックポイント・評価の上書き） ----


def test_build_policy_and_checkpoint_roundtrip(tmp_path):
    """設定から FlowMatchingPolicy を作れ、保存して読み直したモデルが同じ予測を返す。"""
    cfg = TrainConfig(
        policy="flow_matching", chunk_size=K, solver="heun", num_inference_steps=5, **SMALL
    )
    policy = build_policy(cfg, pretrained_backbone=False).eval()
    assert isinstance(policy, FlowMatchingPolicy)
    assert (policy.solver, policy.num_inference_steps) == ("heun", 5)
    policy.action_norm.fit(torch.randn(100, 14) * 3 + 1)
    save_checkpoint(tmp_path / "ckpt.pt", policy, cfg, step=1)
    restored, _ = load_checkpoint(tmp_path / "ckpt.pt")
    batch = fake_batch()
    torch.manual_seed(0)  # 出発点 x_1 を揃える
    a = policy.predict_chunk(batch)
    torch.manual_seed(0)
    b = restored.predict_chunk(batch)
    assert torch.allclose(a, b, atol=1e-5)


def test_eval_solver_overrides():
    """評価の --solver / --num-inference-steps で、学習し直さずにソルバーを切り替えられる。

    Diffusion 用の --sampler / --x0-clip を Flow Matching に、--solver を Diffusion に指定するとエラーにする。
    """
    from pai_act.exps.evaluate import apply_sampler_overrides
    from pai_act.policies.diffusion import DiffusionPolicy

    policy = small_policy()
    apply_sampler_overrides(policy, EvalConfig(ckpt=Path("x"), solver="heun"))
    assert (policy.solver, policy.num_inference_steps) == ("heun", 10)
    apply_sampler_overrides(policy, EvalConfig(ckpt=Path("x"), num_inference_steps=5))
    assert (policy.solver, policy.num_inference_steps) == ("heun", 5)
    for wrong in (
        EvalConfig(ckpt=Path("x"), sampler="ddpm"),
        EvalConfig(ckpt=Path("x"), x0_clip=2.0),
    ):
        with pytest.raises(ValueError):
            apply_sampler_overrides(policy, wrong)
    with pytest.raises(ValueError):
        apply_sampler_overrides(
            DiffusionPolicy(chunk_size=K, pretrained_backbone=False, **SMALL),
            EvalConfig(ckpt=Path("x"), solver="euler"),
        )
