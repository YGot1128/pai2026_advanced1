"""data/ の自作部品（正規化・学習サンプルの切り出し）のテスト。"""

from types import SimpleNamespace

import numpy as np
import torch

from pai_act.data.chunk_dataset import ChunkDataset
from pai_act.data.normalize import Normalizer


def test_normalizer_roundtrip():
    """Normalizer が平均0・標準偏差1に正規化し、denormalize で元の値に戻せることを確かめる。

    ほとんど動かない次元（std ≈ 0）でも std の下限で 0 割りを避けること、統計量が
    state_dict に入り、チェックポイントと一緒に保存されることも確かめる。
    """
    x = np.random.default_rng(0).normal(loc=3.0, scale=2.0, size=(1000, 4)).astype(np.float32)
    x[:, 3] = 1.0  # 動かない次元
    norm = Normalizer(4)
    norm.fit(x)
    z = norm.normalize(torch.from_numpy(x))
    assert torch.allclose(z[:, :3].mean(0), torch.zeros(3), atol=1e-4)
    assert torch.allclose(z[:, :3].std(0), torch.ones(3), atol=1e-3)
    assert norm.std[3] == 1e-2  # std の下限が効いて 0 で割らない
    assert torch.allclose(norm.denormalize(z), torch.from_numpy(x), atol=1e-4)
    # 統計量は state_dict に含まれる（チェックポイントに保存される）
    assert set(norm.state_dict()) == {"mean", "std"}


def fake_source():
    """2エピソード（長さ 5 と 3）。action[t] = t なので、どのフレームを取ったか値で分かる。"""
    n = 8
    return SimpleNamespace(
        images=np.zeros((n, 4, 6, 3), dtype=np.uint8),
        state=np.arange(n, dtype=np.float32)[:, None].repeat(14, axis=1),
        action=np.arange(n, dtype=np.float32)[:, None].repeat(14, axis=1),
        ranges=[(0, 5), (5, 8)],
    )


def test_chunk_dataset_padding():
    """ChunkDataset が時刻 t から K ステップ分の行動を切り出し、足りない分を埋め草にする。

    エピソードの終わりを越える分は最後の行動で埋めて action_is_pad = True にすること、
    隣のエピソードの行動が混ざらないこと、画像を (C, H, W) の uint8 で返すことを確かめる。
    """
    ds = ChunkDataset(fake_source(), episodes=[0, 1], chunk_size=4)
    assert len(ds) == 8

    s = ds[0]  # エピソード0 の t=0: action 0,1,2,3 がすべて有効
    assert s["image"].shape == (3, 4, 6) and s["image"].dtype == torch.uint8
    assert s["action"][:, 0].tolist() == [0, 1, 2, 3]
    assert not s["action_is_pad"].any()

    s = ds[3]  # エピソード0 の t=3: 有効なのは 3,4 だけ。残りは最後の行動 4 で埋める
    assert s["action"][:, 0].tolist() == [3, 4, 4, 4]
    assert s["action_is_pad"].tolist() == [False, False, True, True]

    s = ds[5]  # エピソード1 の先頭（全体のフレーム 5）: 隣のエピソードの行動が混ざらない
    assert s["action"][:, 0].tolist() == [5, 6, 7, 7]
    assert s["state"][0] == 5


def test_chunk_dataset_episode_subset():
    """episodes で選んだエピソードのフレームだけがサンプルになる（学習・検証の分割に使う）。"""
    ds = ChunkDataset(fake_source(), episodes=[1], chunk_size=2)
    assert len(ds) == 3
    assert ds[0]["state"][0] == 5
