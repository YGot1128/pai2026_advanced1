"""ポリシーの入出力の形と、チェックポイントの保存・復元のテスト（CPU・小さい画像で動かす）。"""

import torch

from pai_act.exps.checkpoint import build_policy, load_checkpoint, save_checkpoint
from pai_act.exps.config import TrainConfig, parse_config
from pai_act.policies.base import masked_l1


def fake_batch(b=2, k=5, h=64, w=96):
    """学習ループが渡すのと同じキー・形状・dtype のランダムなバッチを作る（埋め草なし）。"""
    return {
        "image": torch.randint(0, 256, (b, 3, h, w), dtype=torch.uint8),
        "state": torch.randn(b, 14),
        "action": torch.randn(b, k, 14),
        "action_is_pad": torch.zeros(b, k, dtype=torch.bool),
    }


def test_masked_l1_ignores_pad():
    """masked_l1 が埋め草のステップを除いて、有効なステップだけで平均を取ることを確かめる。

    埋め草を含めて平均すると、値 100 の埋め草が損失を支配してしまう。
    """
    pred = torch.zeros(1, 3, 2)
    target = torch.tensor([[[1.0, 1.0], [2.0, 2.0], [100.0, 100.0]]])
    is_pad = torch.tensor([[False, False, True]])
    assert masked_l1(pred, target, is_pad).item() == 1.5


def test_baseline_shapes_and_backward():
    """ベースラインが Policy の約束（スカラーの損失・(B, K, 14) の予測）を守り、逆伝播できる。"""
    cfg = TrainConfig(chunk_size=5, hidden_dim=32)
    policy = build_policy(cfg, pretrained_backbone=False)
    batch = fake_batch(k=5)
    out = policy.compute_loss(batch)
    out["loss"].backward()
    assert out["loss"].ndim == 0
    assert policy.predict_chunk(batch).shape == (2, 5, 14)


def test_checkpoint_roundtrip(tmp_path):
    """チェックポイントから設定とモデルを復元でき、復元したモデルが同じ予測を返すことを確かめる。"""
    cfg = TrainConfig(chunk_size=5, hidden_dim=32)
    policy = build_policy(cfg, pretrained_backbone=False).eval()
    policy.action_norm.fit(torch.randn(100, 14) * 3 + 1)
    save_checkpoint(tmp_path / "ckpt.pt", policy, cfg, step=7)

    restored, restored_cfg = load_checkpoint(tmp_path / "ckpt.pt")
    assert restored_cfg == cfg
    batch = fake_batch(k=5)
    # 正規化の統計量も含めて復元されるので、同じ入力に同じ出力を返す
    assert torch.allclose(policy.predict_chunk(batch), restored.predict_chunk(batch))


def test_parse_config():
    """コマンドライン引数（ケバブケース・--no-xxx の真偽値・指数表記）が TrainConfig に入る。"""
    cfg = parse_config(TrainConfig, ["--chunk-size", "50", "--no-amp", "--lr", "3e-4"])
    assert cfg.chunk_size == 50 and cfg.amp is False and cfg.lr == 3e-4
