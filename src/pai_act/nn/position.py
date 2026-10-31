"""sinusoidal 位置エンコーディング（1D: 系列用 / 2D: 画像特徴用）。

Attention はトークンの順番を区別しない（入力トークンを並べ替えると、出力も同じように並べ替わるだけ）。
そこで「何番目のトークンか」「画像のどこのトークンか」をベクトルで表して与え、位置の情報を持たせる。

1D（Transformer 原著, Vaswani et al. 2017）: 位置 p を dim 次元のベクトルにする。
    pe[p, 2i]   = sin(p / 10000^(2i / dim))
    pe[p, 2i+1] = cos(p / 10000^(2i / dim))      (i = 0, 1, ..., dim/2 - 1)
i が小さい成分は速く振動し、大きい成分はゆっくり振動する。時計の秒針・分針・時針のように、
周期の違う波を組み合わせることで位置を一意に表せる。学習するパラメータはない。

2D（DETR / ACT）: dim の前半で縦の位置 y、後半で横の位置 x を、それぞれ dim/2 次元の 1D と同じ式で表して連結する。
位置は画素の番号そのままではなく (0, 2π] に正規化する（特徴マップの大きさが変わっても値の範囲が揃う）:
    y = (row + 1) / H * 2π,   x = (col + 1) / W * 2π

ACT での使い方:
    - 2D: ResNet18 の特徴マップ 15x20 の各位置 → 画像トークン 300 個の位置（policies/trunk.py）
    - 1D: CVAE エンコーダの入力列 [CLS, 関節角, 行動 k 個] の位置（policies/act.py）
    - それ以外（Decoder の行動クエリ、z・関節角トークン）の位置は、sinusoidal ではなく学習するベクトルを使う
"""

import math

import torch


def sinusoidal_encoding(
    positions: torch.Tensor, dim: int, temperature: float = 10000.0
) -> torch.Tensor:
    """任意の（実数の）位置の列を sinusoidal ベクトルにする。1D と 2D の共通部品。

    Args:
        positions: (N,) float。各トークンの位置。
        dim: 出力の次元数（偶数）。
        temperature: 上の式の 10000。

    Returns:
        (N, dim)。偶数番目の成分が sin、奇数番目の成分が cos（上の式どおり、sin と cos が交互に並ぶ）。
    """
    # Step 1: dim が偶数であることを assert で確認する
    assert dim % 2 == 0, "dim must be even"

    # Step 2: 各 i (= 0, 1, ..., dim/2 - 1) の周波数 1 / temperature^(2i / dim) を作る。形は (dim/2,)
    exponent = (
        torch.arange(0, dim, 2, dtype=positions.dtype, device=positions.device) / dim
    )  # (dim/2,)
    freqs = 1.0 / (temperature**exponent)  # (dim/2,)

    # Step 3: 角度 = 位置 × 周波数 を、全ての (位置, 周波数) の組について作る。形は (N, dim/2)
    angles = positions[:, None] * freqs[None, :]  # (N, dim/2)

    # Step 4: 出力 (N, dim) を用意し、偶数番目の列に sin(角度)、奇数番目の列に cos(角度) を入れる
    pe = torch.zeros(
        positions.shape[0], dim, device=positions.device, dtype=angles.dtype
    )  # (N, dim)
    pe[:, 0::2] = torch.sin(angles)
    pe[:, 1::2] = torch.cos(angles)

    return pe


def sinusoidal_1d(length: int, dim: int, device: torch.device | str | None = None) -> torch.Tensor:
    """位置 0, 1, ..., length-1 の 1D 位置エンコーディング。

    Returns:
        (length, dim)
    """
    # Step 1: 位置 0, 1, ..., length-1 を float のテンソルで作る（device を指定する）
    positions = torch.arange(length, dtype=torch.float32, device=device)

    # Step 2: sinusoidal_encoding に渡して返す
    return sinusoidal_encoding(positions, dim)


def sinusoidal_2d(
    height: int, width: int, dim: int, device: torch.device | str | None = None
) -> torch.Tensor:
    """height x width の格子の 2D 位置エンコーディング。

    行優先（row-major）の順に並べる: 位置 (row, col) は index row * width + col。
    これは特徴マップ (B, C, H, W) を flatten(2) したときのトークンの並び順と一致する。

    Returns:
        (height * width, dim)。前半 dim/2 が縦の位置、後半 dim/2 が横の位置を表す。
    """
    # Step 1: dim が 4 の倍数であることを assert で確認する（前半・後半それぞれが sin/cos の組になるため）
    assert dim % 4 == 0, "dim must be divisible by 4"

    # Step 2: 縦の位置 y = (row + 1) / height * 2π を row = 0..height-1 について作る → (height,)
    #         横の位置 x = (col + 1) / width  * 2π を col = 0..width-1  について作る → (width,)
    row = torch.arange(height, dtype=torch.float32, device=device)  # (height,)
    y = (row + 1) / height * 2 * math.pi  # (height,)
    col = torch.arange(width, dtype=torch.float32, device=device)  # (width,)
    x = (col + 1) / width * 2 * math.pi  # (width,)

    # Step 3: それぞれを dim/2 次元でエンコードする → y_enc (height, dim/2), x_enc (width, dim/2)
    y_enc = sinusoidal_encoding(y, dim // 2)  # (height, dim/2)
    x_enc = sinusoidal_encoding(x, dim // 2)  # (width, dim/2)

    # Step 4: (height, width, dim) の格子に広げる。
    # 位置 (row, col) のベクトルは[y_enc[row] と x_enc[col] の連結] にする
    pe = torch.cat(
        [y_enc[:, None, :].expand(height, width, -1), x_enc[None, :, :].expand(height, width, -1)],
        dim=-1,
    )

    # Step 5: (height * width, dim) に reshape して返す
    return pe.reshape(height * width, dim)
