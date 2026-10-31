"""仮モデル（baseline）: ResNet18 + MLP で行動チャンクを直接回帰する。

パイプライン（学習 → ロールアウト → 動画保存）を ACT より先に一通り繋げて検証するためのモデル。
Transformer も CVAE も使わない最小構成で、ACT と同じインタフェース（policies/base.py）を実装する。

    image (B,3,480,640) ─ ResNet18 ─ (B,512,15,20) ─ 空間平均 ─ (B,512) ─┐
                                                                        concat ─ MLP ─ (B, k*14) ─ reshape ─ (B,k,14)
    state (B,14) ─────────────────────────────── 正規化 ─────── (B,14) ─┘

ACT との違い（= ACT で改善される点）:
    - 空間平均で「どこに何があるか」の位置情報がかなり失われる
      → ACT は 15x20 の特徴を位置エンコーディング付きのトークンとして Transformer で扱う
    - 決定的な回帰なので、人間のデモに含まれる「同じ状況でも動きが揺らぐ」多峰性を平均してしまう
      → ACT は CVAE の潜在変数でデモのスタイルの違いを表す
"""

import torch
from torch import nn

from pai_act.nn.backbone import ResNet18Backbone
from pai_act.policies.base import ACTION_DIM, STATE_DIM, Policy, masked_l1


class BaselinePolicy(Policy):
    """ResNet18 の大域特徴 + 関節角から、MLP で k ステップ分の行動を一度に出す。

    Args:
        chunk_size: 行動チャンクの長さ k。
        hidden_dim: MLP の中間層の幅。
        pretrained_backbone: ResNet18 に ImageNet 事前学習重みを使うか。
    """

    def __init__(self, chunk_size: int, hidden_dim: int = 512, pretrained_backbone: bool = True):
        super().__init__(chunk_size)
        self.backbone = ResNet18Backbone(pretrained=pretrained_backbone)
        self.head = nn.Sequential(
            nn.Linear(ResNet18Backbone.out_channels + STATE_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, chunk_size * ACTION_DIM),
        )

    def _forward(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """正規化済みの行動チャンク (B, k, 14) を返す（損失計算と推論で共通の部分）。"""
        feat = self.backbone(image).mean(dim=(2, 3))  # (B, 512, h, w) → (B, 512)
        x = torch.cat([feat, self.state_norm.normalize(state)], dim=-1)
        return self.head(x).view(-1, self.chunk_size, ACTION_DIM)

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        pred = self._forward(batch["image"], batch["state"])
        # 目標値も正規化してから比べる（14次元の誤差のスケールを揃えるため）
        target = self.action_norm.normalize(batch["action"])
        l1 = masked_l1(pred, target, batch["action_is_pad"])
        return {"loss": l1, "l1": l1.detach()}

    @torch.no_grad()
    def predict_chunk(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.action_norm.denormalize(self._forward(obs["image"], obs["state"]))
