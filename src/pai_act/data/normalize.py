"""観測（関節角）と行動の正規化。

ニューラルネットワークは入出力のスケールが揃っている方が学習しやすい。
ALOHA の 14 次元ベクトルは、腕の関節角（rad, 数値の幅は ±1 程度）とグリッパー開度（0〜1 程度）で
スケールも平均も異なるので、学習データの平均と標準偏差で次元ごとに標準化する:

    x_norm = (x - mean) / std        （正規化: ニューラルネットワークへの入力、学習の目標値）
    x      = x_norm * std + mean     （逆変換: ニューラルネットワークの出力をシミュレータに渡す行動に戻す）

統計量は nn.Module のバッファとして持たせる。こうすると
- ポリシーの state_dict に統計量が含まれ、チェックポイント1つで推論まで再現できる
- policy.to("cuda") で統計量も一緒に GPU に載る
- バッファは学習対象のパラメータではないので optimizer に更新されない
"""

import numpy as np
import torch
from torch import nn


class Normalizer(nn.Module):
    """次元ごとの平均・標準偏差で標準化する。

    作成直後は mean=0, std=1（恒等変換）。学習前に fit() で学習データの統計量を設定する。
    チェックポイントから読み込む場合は load_state_dict() で上書きされるので fit() は不要。

    Args:
        dim: ベクトルの次元数（ALOHA なら 14）。
    """

    # register_buffer で作る属性は、型チェッカーに Tensor だと伝えるために宣言しておく
    mean: torch.Tensor
    std: torch.Tensor

    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))

    @torch.no_grad()
    def fit(self, x: np.ndarray | torch.Tensor, min_std: float = 1e-2) -> None:
        """学習データ x: (N, dim) から統計量を求めてバッファに書き込む。

        ほとんど動かない次元（std ≈ 0）で割ると値が爆発するので、std に下限 min_std を設ける。
        """
        x = torch.as_tensor(x, dtype=torch.float32)
        self.mean.copy_(x.mean(dim=0))
        self.std.copy_(x.std(dim=0).clamp(min=min_std))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., dim) を標準化する。先頭の次元はいくつあってもよい（ブロードキャスト）。"""
        return (x - self.mean) / self.std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """normalize() の逆変換。"""
        return x * self.std + self.mean
