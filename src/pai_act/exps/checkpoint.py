"""ポリシーの組み立てと、チェックポイントの保存・読み込み。

チェックポイント（.pt）には次を1つにまとめて保存する:
    config:     学習時の TrainConfig（dict）… どのポリシーをどの大きさで作ったかが分かる
    step:       何ステップ学習した時点か
    state_dict: ポリシーの全パラメータとバッファ（正規化の統計量を含む）

評価時は config からポリシーを同じ形で作り直し、state_dict を流し込めば学習時と同じものが復元できる。
"""

from dataclasses import asdict
from pathlib import Path

import torch

from pai_act.exps.config import TrainConfig
from pai_act.policies.act import ACTPolicy
from pai_act.policies.base import Policy
from pai_act.policies.baseline import BaselinePolicy
from pai_act.policies.diffusion import DiffusionPolicy


def build_policy(cfg: TrainConfig, pretrained_backbone: bool = True) -> Policy:
    """設定に従ってポリシーを作る。新しい手法を足すときはここに分岐を追加する。"""
    if cfg.policy == "baseline":
        return BaselinePolicy(
            chunk_size=cfg.chunk_size,
            hidden_dim=cfg.hidden_dim,
            pretrained_backbone=pretrained_backbone,
        )
    if cfg.policy == "act":
        return ACTPolicy(
            chunk_size=cfg.chunk_size,
            d_model=cfg.d_model,
            num_heads=cfg.num_heads,
            dim_feedforward=cfg.dim_feedforward,
            num_encoder_layers=cfg.num_encoder_layers,
            num_decoder_layers=cfg.num_decoder_layers,
            num_vae_encoder_layers=cfg.num_vae_encoder_layers,
            latent_dim=cfg.latent_dim,
            dropout=cfg.dropout,
            kl_weight=cfg.kl_weight,
            learned_query_pos=cfg.learned_query_pos,
            pretrained_backbone=pretrained_backbone,
        )
    if cfg.policy == "diffusion":
        return DiffusionPolicy(
            chunk_size=cfg.chunk_size,
            d_model=cfg.d_model,
            num_heads=cfg.num_heads,
            dim_feedforward=cfg.dim_feedforward,
            num_encoder_layers=cfg.num_encoder_layers,
            num_decoder_layers=cfg.num_decoder_layers,
            dropout=cfg.dropout,
            num_diffusion_steps=cfg.num_diffusion_steps,
            sampler=cfg.sampler,
            num_inference_steps=cfg.num_inference_steps,
            pretrained_backbone=pretrained_backbone,
        )
    raise ValueError(f"unknown policy: {cfg.policy}")


def save_checkpoint(path: Path, policy: Policy, cfg: TrainConfig, step: int) -> None:
    torch.save({"config": asdict(cfg), "step": step, "state_dict": policy.state_dict()}, path)


def load_checkpoint(path: Path, device: str | torch.device = "cpu") -> tuple[Policy, TrainConfig]:
    """チェックポイントからポリシーを復元し、評価モードにして返す。"""
    ckpt = torch.load(path, map_location=device)
    cfg = TrainConfig(**ckpt["config"])
    # 重みは state_dict で上書きするので、事前学習重みのダウンロードは不要
    policy = build_policy(cfg, pretrained_backbone=False)
    policy.load_state_dict(ckpt["state_dict"])
    return policy.to(device).eval(), cfg
