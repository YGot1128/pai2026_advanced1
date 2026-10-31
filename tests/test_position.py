"""nn/position.py のテスト。

uv run pytest tests/test_position.py -v
"""

import math

import torch

from pai_act.nn.position import sinusoidal_1d, sinusoidal_2d, sinusoidal_encoding


def test_encoding_formula():
    """sinusoidal_encoding が Transformer 原著の式どおりの値になることを、列ごとに確かめる。

    列 2i が sin(p / 10000^(2i/dim))、列 2i+1 が同じ周波数の cos。sin と cos を交互に並べる
    （前半 sin・後半 cos ではない）ことと、周波数の指数の分子が 2i であることがポイント。
    """
    dim = 8
    pos = torch.tensor([0.0, 1.0, 2.5, 7.0])
    pe = sinusoidal_encoding(pos, dim)
    assert pe.shape == (4, dim)
    # i = 0 の組: 周波数 1 → sin(p), cos(p)
    assert torch.allclose(pe[:, 0], torch.sin(pos))
    assert torch.allclose(pe[:, 1], torch.cos(pos))
    # i = 1 の組: 周波数 1 / 10000^(2/8)
    assert torch.allclose(pe[:, 2], torch.sin(pos / 10000 ** (2 / dim)))
    assert torch.allclose(pe[:, 3], torch.cos(pos / 10000 ** (2 / dim)))
    # 最後の組（i = dim/2 - 1）
    assert torch.allclose(pe[:, 7], torch.cos(pos / 10000 ** (6 / dim)))


def test_encoding_sin_cos_pairs():
    """隣り合う (sin, cos) の組が同じ周波数を使っていることを、sin² + cos² = 1 で確かめる。

    組ごとに周波数がずれていると 1 にならない。式を1つずつ比べる前のテストより粗いが、
    次元が大きいときにも全部の組をまとめて確かめられる。
    """
    pe = sinusoidal_encoding(torch.arange(10).float(), 16)
    # 各 (sin, cos) の組は単位円の上にある
    assert torch.allclose(pe[:, 0::2] ** 2 + pe[:, 1::2] ** 2, torch.ones(10, 8))


def test_1d():
    """sinusoidal_1d が位置 0, 1, ..., n-1 のエンコーディングを (n, dim) で返すことを確かめる。

    位置ごとにベクトルが違うことも確かめる。同じベクトルの位置があると、モデルはそれらの
    位置を区別できない。
    """
    pe = sinusoidal_1d(102, 32)
    assert pe.shape == (102, 32)
    assert torch.allclose(pe, sinusoidal_encoding(torch.arange(102).float(), 32))
    # 位置が違えばベクトルも違う
    assert len(torch.unique(pe, dim=0)) == 102


def test_2d_shape_and_values():
    """sinusoidal_2d の形状と、左上・右下の画素での値を確かめる。

    DETR に合わせ、行・列の番号を (index + 1) / 個数 × 2π に正規化してから 1D の式に入れる。
    前半 dim/2 次元が縦（行）、後半 dim/2 次元が横（列）の位置を表す。
    """
    h, w, dim = 3, 4, 16
    pe = sinusoidal_2d(h, w, dim)
    assert pe.shape == (h * w, dim)
    # 左上 (row=0, col=0): 縦の位置 y = 1/h * 2π、横の位置 x = 1/w * 2π
    assert torch.isclose(pe[0, 0], torch.tensor(math.sin(2 * math.pi / h)))
    assert torch.isclose(pe[0, dim // 2], torch.tensor(math.sin(2 * math.pi / w)))
    # 右下: y = x = 2π
    assert torch.isclose(pe[-1, 0], torch.tensor(math.sin(2 * math.pi)), atol=1e-6)
    assert torch.isclose(pe[-1, 1], torch.tensor(1.0))
    assert len(torch.unique(pe, dim=0)) == h * w


def test_2d_row_major_layout():
    """sinusoidal_2d の行が、画素を行優先（index = row * w + col）で並べた順になっている。

    画像の特徴マップを flatten したトークンの順番と一致していないと、各トークンに別の画素の
    位置が付いてしまう。値の形は正しくても気づきにくい誤りなので、並び方だけを別に確かめる。
    """
    h, w, dim = 3, 4, 16
    grid = sinusoidal_2d(h, w, dim).view(h, w, dim)  # index row*w + col → [row, col]
    y_part, x_part = grid[..., : dim // 2], grid[..., dim // 2 :]
    # 前半（縦の位置）は行だけで決まり、後半（横の位置）は列だけで決まる
    assert torch.allclose(y_part, y_part[:, :1].expand(h, w, -1))
    assert torch.allclose(x_part, x_part[:1, :].expand(h, w, -1))
    # 前半・後半それぞれが dim/2 次元の 1D エンコーディングになっている
    rows = (torch.arange(h).float() + 1) / h * 2 * math.pi
    assert torch.allclose(y_part[:, 0], sinusoidal_encoding(rows, dim // 2))
