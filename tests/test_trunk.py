"""policies/trunk.py のテスト。

uv run pytest tests/test_trunk.py -v
"""

import torch

from pai_act.nn.position import sinusoidal_2d
from pai_act.policies.trunk import ObservationTrunk

B, C, H, W, D = 2, 512, 3, 4, 32


def test_shapes():
    """ObservationTrunk が「関節角1トークン + 画像 H*W トークン」の列と、その位置を返す。

    位置エンコーディングはバッチで共有するので、バッチの次元は 1 になる。
    """
    trunk = ObservationTrunk(D)
    tokens, pos = trunk(torch.randn(B, C, H, W), torch.randn(B, 14))
    assert tokens.shape == (B, 1 + H * W, D)
    assert pos.shape == (1, 1 + H * W, D)


def test_token_order_and_pos():
    """トークンの並び順と、各トークンに付く位置エンコーディングが正しいことを確かめる。

    形状だけでは、flatten の順番の誤り（列優先になっている、チャネルと位置が入れ替わっている）や、
    位置エンコーディングが別のトークンに付いている誤りは見つからない。そこで特定の画素の特徴から
    作ったトークンが、行優先で計算した位置にあるかまで確かめる。
    """
    trunk = ObservationTrunk(D)
    feature, state = torch.randn(B, C, H, W), torch.randn(B, 14)
    tokens, pos = trunk(feature, state)
    # 先頭は関節角トークン: 関節角だけで決まる
    assert torch.allclose(tokens[:, 0], trunk.state_proj(state))
    # 残りは画像トークン（行優先の順）: 位置 (row, col) のトークンはその画素の特徴だけで決まる
    r, c = 1, 2
    pixel = feature[:, :, r, c][:, :, None, None]
    assert torch.allclose(tokens[:, 1 + r * W + c], trunk.image_proj(pixel)[:, :, 0, 0], atol=1e-5)
    # 位置: 関節角トークンは学習するベクトル、画像トークンは 2D sinusoidal
    assert torch.equal(pos[0, 0], trunk.state_pos.view(-1))
    assert torch.allclose(pos[0, 1:], sinusoidal_2d(H, W, D))
    assert trunk.state_pos.requires_grad
