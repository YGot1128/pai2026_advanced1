"""policies/denoiser.py のテスト（CPU・小さいモデル・小さい画像で動かす）。

uv run pytest tests/test_denoiser.py -v

DenoiserPolicy は compute_loss / predict_chunk を持たない基底クラスなので、テストの中で
中身が空のサブクラスを作って試す（Diffusion / Flow Matching はこの2つだけを足す）。
"""

from typing import Any

import torch
from torch import nn

from pai_act.nn.backbone import ResNet18Backbone
from pai_act.nn.position import sinusoidal_encoding
from pai_act.policies.denoiser import DenoiserPolicy

K = 5
D = 32
SMALL: dict[str, Any] = {
    "d_model": D,
    "num_heads": 4,
    "dim_feedforward": 64,
    "num_encoder_layers": 1,
    "num_decoder_layers": 1,
    "dropout": 0.0,
}
# 64x96 の画像は ResNet18（縮小率 32）で 2x3 の特徴マップになる → memory は関節角 1 + 画像 6 トークン
H, W = 64, 96
NUM_MEMORY = 1 + (H // 32) * (W // 32)


class _Denoiser(DenoiserPolicy):
    """テスト用: 学習・推論のメソッドは使わないので空にしておく。"""

    def compute_loss(self, batch):
        raise AssertionError("使わない")

    def predict_chunk(self, obs):
        raise AssertionError("使わない")


def small_policy() -> _Denoiser:
    torch.manual_seed(0)
    return _Denoiser(chunk_size=K, pretrained_backbone=False, **SMALL).eval()


def fake_obs(b=2):
    torch.manual_seed(1)
    image = torch.randint(0, 256, (b, 3, H, W), dtype=torch.uint8)
    return image, torch.randn(b, 14)


def test_backbone_attribute():
    """画像バックボーンが self.backbone にある（学習ループがパラメータ名 "backbone." で学習率を分けるため）。"""
    policy = small_policy()
    assert isinstance(policy.backbone, ResNet18Backbone)
    assert any(name.startswith("backbone.") for name, _ in policy.named_parameters())


def test_encode_obs_shapes():
    """_encode_obs が memory (B, 1+h*w, d) と、その位置 (1, 1+h*w, d) を返す。"""
    policy = small_policy()
    memory, memory_pos = policy._encode_obs(*fake_obs())
    assert memory.shape == (2, NUM_MEMORY, D)
    assert memory_pos.shape == (1, NUM_MEMORY, D)


def test_embed_time():
    """時刻の埋め込みは「sinusoidal_encoding → time_mlp」で、形は (B, 1, d)。

    整数の t（Diffusion）でも実数の t（Flow Matching）でも受け取れる。
    sinusoidal を挟まずに t をそのまま MLP に入れると、t の小さな違いを区別しにくい。
    """
    policy = small_policy()
    t = torch.tensor([0, 7, 99])
    emb = policy._embed_time(t)
    assert emb.shape == (3, 1, D)
    time_mlp = policy.time_mlp
    assert isinstance(time_mlp, nn.Module)
    expected = time_mlp(sinusoidal_encoding(t.float(), D)).unsqueeze(1)
    assert torch.allclose(emb, expected, atol=1e-6)
    # 実数の t も受け取れる
    assert policy._embed_time(torch.tensor([0.25, 0.5])).shape == (2, 1, D)


def test_denoise_shape_and_inputs():
    """_denoise が (B, k, 14) を返し、x_t・t・memory のどれを変えても出力が変わる。

    どれかを変えても出力が変わらないなら、その入力を forward で使い忘れている。
    """
    policy = small_policy()
    memory, memory_pos = policy._encode_obs(*fake_obs())
    torch.manual_seed(2)
    x_t = torch.randn(2, K, 14)
    t = torch.tensor([3, 50])
    out = policy._denoise(x_t, t, memory, memory_pos)
    assert out.shape == (2, K, 14)

    def changed(a, b):
        return not torch.allclose(a, b, atol=1e-5)

    assert changed(out, policy._denoise(x_t + 1.0, t, memory, memory_pos)), "x_t を使っていない"
    assert changed(out, policy._denoise(x_t, t + 10, memory, memory_pos)), "t を使っていない"
    assert changed(out, policy._denoise(x_t, t, memory + 1.0, memory_pos)), "memory を使っていない"


def test_query_pos_distinguishes_steps():
    """x_t が全ステップで同じでも、出力はステップごとに違う（クエリの位置ベクトルで区別している）。

    ACT と違い Decoder の入力は x_t になったが、x_t だけでは「何ステップ目の行動か」が分からない
    （同じ値の行動が並ぶこともある）。query_pos を使い忘れると、全ステップが同じ出力になる。
    """
    policy = small_policy()
    assert policy.query_pos.shape == (1, K, D)
    memory, memory_pos = policy._encode_obs(*fake_obs(1))
    out = policy._denoise(torch.zeros(1, K, 14), torch.tensor([10]), memory, memory_pos)
    assert not torch.allclose(out[0, 0], out[0, 1], atol=1e-5)


def test_grads_reach_all_params():
    """観測の符号化からデノイザの出力まで、全パラメータに勾配が流れる（使い忘れた層がない）。"""
    policy = small_policy().train()
    memory, memory_pos = policy._encode_obs(*fake_obs())
    out = policy._denoise(torch.randn(2, K, 14), torch.tensor([1, 2]), memory, memory_pos)
    out.pow(2).sum().backward()
    for name, p in policy.named_parameters():
        assert p.grad is not None, f"{name} に勾配が流れていない（使われていない層がある？）"
