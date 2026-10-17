"""policies/act.py のテスト（CPU・小さいモデル・小さい画像で動かす）。

uv run pytest tests/test_act.py -v
"""

import math
from typing import Any

import torch
from torch.distributions import Normal, kl_divergence

from pai_act.exps.checkpoint import build_policy, load_checkpoint, save_checkpoint
from pai_act.exps.config import TrainConfig
from pai_act.nn.position import sinusoidal_1d
from pai_act.policies.act import ACTPolicy, kl_to_standard_normal, reparameterize

K = 5
# int と float が混ざる。Any を付けないと値が全部 float とみなされ、int の引数に渡すところで型エラーになる
SMALL: dict[str, Any] = {
    "d_model": 32,
    "num_heads": 4,
    "dim_feedforward": 64,
    "num_encoder_layers": 1,
    "num_decoder_layers": 1,
    "num_vae_encoder_layers": 1,
    "latent_dim": 8,
    "dropout": 0.0,
}


def small_policy(**overrides) -> ACTPolicy:
    """CPU でもすぐ動く小さい ACTPolicy を作る。事前学習重みはダウンロードしない。"""
    torch.manual_seed(0)
    return ACTPolicy(chunk_size=K, pretrained_backbone=False, **{**SMALL, **overrides})


def fake_batch(b=2, h=64, w=96):
    """学習ループが渡すのと同じキー・形状・dtype のランダムなバッチを作る（埋め草なし）。"""
    torch.manual_seed(1)
    return {
        "image": torch.randint(0, 256, (b, 3, h, w), dtype=torch.uint8),
        "state": torch.randn(b, 14),
        "action": torch.randn(b, K, 14),
        "action_is_pad": torch.zeros(b, K, dtype=torch.bool),
    }


# ---- reparameterize / KL ----


def test_reparameterize_statistics():
    """reparameterize のサンプルが平均 μ・標準偏差 σ の正規分布に従う（多数のサンプルで確かめる）。

    log σ² から σ を出すときに 0.5 を掛け忘れる（σ² を σ として使う）誤りがここで見つかる。
    """
    torch.manual_seed(0)
    mu = torch.full((100000, 1), 1.0)
    logvar = torch.full((100000, 1), math.log(4.0))  # σ = 2
    z = reparameterize(mu, logvar)
    assert z.shape == mu.shape
    assert abs(z.mean().item() - 1.0) < 0.05
    assert abs(z.std().item() - 2.0) < 0.05


def test_reparameterize_passes_grad():
    """サンプル z から μ と log σ² の両方へ勾配が流れることを確かめる。

    これが reparameterization trick の目的。torch.normal(mu, std) のように分布から直接
    サンプルすると勾配が切れ、CVAE エンコーダが学習されない。
    """
    mu = torch.zeros(4, 3, requires_grad=True)
    logvar = torch.zeros(4, 3, requires_grad=True)
    reparameterize(mu, logvar).sum().backward()
    assert mu.grad is not None and logvar.grad is not None


def test_kl():
    """KL(N(μ, σ²) || N(0, 1)) の値を、torch.distributions で計算した値と比べて確かめる。

    潜在変数の次元で和を取り、バッチで平均する（ACT 公式実装と同じ集約）。μ = 0・log σ² = 0
    （事前分布そのもの）のときに 0 になることも確かめる。
    """
    zeros = torch.zeros(3, 8)
    assert kl_to_standard_normal(zeros, zeros).item() == 0.0  # 事前分布と一致すれば 0
    torch.manual_seed(0)
    mu, logvar = torch.randn(3, 8), torch.randn(3, 8)
    expected = kl_divergence(Normal(mu, (0.5 * logvar).exp()), Normal(0.0, 1.0)).sum(-1).mean()
    assert torch.allclose(kl_to_standard_normal(mu, logvar), expected, atol=1e-5)


# ---- ACTPolicy ----


def test_compute_loss_and_grad():
    """compute_loss が loss = l1 + kl_weight * kl のスカラーを返し、全パラメータに勾配が流れる。

    勾配が届かないパラメータは、作ったのに forward で使っていない層（CVAE エンコーダの一部や、
    使い忘れた射影など）があることを示す。
    """
    policy = small_policy()
    out = policy.compute_loss(fake_batch())
    assert set(out) >= {"loss", "l1", "kl"}
    assert out["loss"].ndim == 0
    assert torch.allclose(out["loss"], out["l1"] + policy.kl_weight * out["kl"])
    out["loss"].backward()
    for name, p in policy.named_parameters():
        assert p.grad is not None, f"{name} に勾配が流れていない（使われていない層がある？）"


def test_predict_chunk_shape_and_deterministic():
    """predict_chunk が (B, K, 14) の行動を返し、同じ入力には同じ出力を返すことを確かめる。

    推論では CVAE エンコーダを使わず z = 0（事前分布の平均）にする。z をサンプルしていると
    出力が毎回変わってしまう。
    """
    policy = small_policy().eval()
    batch = fake_batch()
    a1 = policy.predict_chunk(batch)
    assert a1.shape == (2, K, 14)
    # 推論は z = 0 なので、同じ入力には同じ出力
    assert torch.equal(a1, policy.predict_chunk(batch))


def test_padded_actions_are_ignored():
    """埋め草（action_is_pad = True）のステップの値を変えても、損失（L1 も KL も）は変わらない。

    L1 では masked_l1 で埋め草を除くこと、KL では CVAE エンコーダの key_padding_mask で
    埋め草の行動トークンを無視すること、の両方ができていないと通らない。
    """
    policy = small_policy().eval()
    batch = fake_batch()
    batch["action_is_pad"][:, -2:] = True
    changed = {**batch, "action": batch["action"].clone()}
    changed["action"][:, -2:] += 100.0
    torch.manual_seed(0)
    out1 = policy.compute_loss(batch)
    torch.manual_seed(0)  # z のサンプルの乱数を揃える
    out2 = policy.compute_loss(changed)
    assert torch.allclose(out1["loss"], out2["loss"], atol=1e-5)


def test_overfits_single_batch():
    """1つのバッチを 200 回学習すると、L1 が最初の 1/4 未満まで下がる（過学習テスト）。

    1バッチだけを繰り返し学習すれば、損失はほぼ 0 まで下がるはず。下がらなければ、どこかで
    勾配が切れている・形を取り違えている可能性が高い。
    バックボーンは固定する: 事前学習重みなしの ResNet は（BatchNorm を固定しているため）値が暴れて
    学習が不安定になり、テストの結果がシード次第になる。ここで確かめたいのは Transformer 側。
    """
    policy = small_policy(kl_weight=1.0)
    policy.backbone.requires_grad_(False)
    batch = fake_batch()
    opt = torch.optim.Adam([p for p in policy.parameters() if p.requires_grad], lr=1e-3)
    first = None
    for _ in range(200):
        out = policy.compute_loss(batch)
        opt.zero_grad()
        out["loss"].backward()
        opt.step()
        first = first if first is not None else out["l1"].item()
    assert out["l1"].item() < 0.25 * first, f"L1 {first:.3f} → {out['l1'].item():.3f}"


def test_build_policy_and_checkpoint_roundtrip(tmp_path):
    """設定から ACTPolicy を作れ、保存して読み直したモデルが同じ予測を返すことを確かめる。

    学習・評価のコードと ACT の配線（config のフィールド名と build_policy）が合っているか、
    正規化の統計量などの buffer もチェックポイントに入っているかを確かめる。
    """
    cfg = TrainConfig(policy="act", chunk_size=K, **SMALL)
    policy = build_policy(cfg, pretrained_backbone=False).eval()
    assert isinstance(policy, ACTPolicy)
    policy.action_norm.fit(torch.randn(100, 14) * 3 + 1)
    save_checkpoint(tmp_path / "ckpt.pt", policy, cfg, step=1)
    restored, _ = load_checkpoint(tmp_path / "ckpt.pt")
    batch = fake_batch()
    assert torch.allclose(policy.predict_chunk(batch), restored.predict_chunk(batch))


def test_query_pos_learned_or_fixed():
    """行動クエリの位置ベクトルが、learned_query_pos で学習版 / 固定版に作り分けられる。

    学習版（既定。ACT 公式実装）はパラメータ、固定版（論文の記述）は sinusoidal_1d の値の
    バッファになる。固定版は config の --no-learned-query-pos から build_policy を通して作り、
    配線も合わせて確かめる。
    """
    learned = small_policy()
    assert "query_pos" in dict(learned.named_parameters())
    assert learned.query_pos.shape == (1, K, SMALL["d_model"])

    cfg = TrainConfig(policy="act", chunk_size=K, learned_query_pos=False, **SMALL)
    fixed = build_policy(cfg, pretrained_backbone=False).eval()
    assert isinstance(fixed, ACTPolicy)
    assert "query_pos" not in dict(fixed.named_parameters()), "固定版が学習の対象になっている"
    expected = sinusoidal_1d(K, SMALL["d_model"]).unsqueeze(0)
    assert torch.allclose(fixed.query_pos, expected)
    assert fixed.predict_chunk(fake_batch()).shape == (2, K, 14)
