"""画像バックボーン: ResNet18（借用部分: ImageNet 事前学習重み）。

カメラ画像を「特徴マップ」に変換する。ResNet18 の最後の全結合層と global average pooling を外し、
畳み込み部分だけを使う。入力 480x640 に対して、空間解像度が 1/32 の特徴マップが出る:

    (B, 3, 480, 640) uint8  →  (B, 512, 15, 20) float

ACT ではこの 15x20=300 個の特徴ベクトルをそれぞれ1トークンとして Transformer に入れる。
仮モデル（policies/baseline.py）では空間方向に平均して 512 次元のベクトル1本にする。

BatchNorm について:
    BatchNorm はミニバッチの統計で正規化し、学習中に running mean/var を更新する層。
    バッチサイズ 8 程度の小さいバッチではこの統計が不安定になるため、DETR / ACT と同じく
    FrozenBatchNorm2d（ImageNet で学習済みの統計とスケールを固定した BatchNorm）に置き換える。
    畳み込みの重みは学習で微調整する（学習率は小さめにする: exps/config.py の lr_backbone）。
"""

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.ops.misc import FrozenBatchNorm2d

# ImageNet の画素値の平均・標準偏差（RGB 順、0〜1 スケール）。事前学習時と同じ正規化をかける必要がある
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ResNet18Backbone(nn.Module):
    """uint8 画像を受け取り、ResNet18 の最終畳み込み層の特徴マップを返す。

    Args:
        pretrained: True なら ImageNet 事前学習重みを読み込む（初回はダウンロードが走る）。
            チェックポイントから復元するときは重みが上書きされるので False でよい。
    """

    out_channels = 512  # ResNet18 の最終段のチャンネル数
    pixel_mean: torch.Tensor  # register_buffer で作る属性（型チェッカー向けの宣言）
    pixel_std: torch.Tensor

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = resnet18(weights=weights, norm_layer=FrozenBatchNorm2d)
        # 分類用の avgpool / fc を除いた畳み込み部分だけを使う
        self.body = nn.Sequential(
            net.conv1, net.bn1, net.relu, net.maxpool,
            net.layer1, net.layer2, net.layer3, net.layer4,
        )  # fmt: skip
        self.register_buffer("pixel_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("pixel_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """image: (B, 3, H, W) uint8 → 特徴マップ (B, 512, H/32, W/32)。"""
        x = image.float() / 255.0
        x = (x - self.pixel_mean) / self.pixel_std
        return self.body(x)
