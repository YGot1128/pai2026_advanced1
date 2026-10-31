"""exps/inference.py の ChunkExecutor（action chunking / temporal ensembling）のテスト。"""

import numpy as np

from pai_act.exps.inference import ChunkExecutor


class FakePolicy:
    """推論のたびに「何回目の推論か × 100 + チャンク内の位置」を値に持つチャンクを返す。"""

    def __init__(self, k: int):
        self.k = k
        self.calls = 0

    def __call__(self) -> np.ndarray:
        chunk = (self.calls * 100 + np.arange(self.k, dtype=float))[:, None]
        self.calls += 1
        return chunk


def test_action_chunking_executes_chunk_in_order():
    """query_every = chunk_size のとき、チャンクを先頭から順に実行し、使い切ったら推論し直す。

    ACT の action chunking の基本形（K ステップに1回だけ推論する）。
    """
    policy = FakePolicy(k=4)
    ex = ChunkExecutor(chunk_size=4, query_every=4)
    actions = [ex.act(policy)[0] for _ in range(8)]
    assert actions == [0, 1, 2, 3, 100, 101, 102, 103]
    assert policy.calls == 2  # 4ステップに1回だけ推論している


def test_query_every_smaller_than_chunk():
    """query_every < chunk_size のとき、query_every ステップごとに新しいチャンクに切り替える。"""
    policy = FakePolicy(k=4)
    ex = ChunkExecutor(chunk_size=4, query_every=2)
    actions = [ex.act(policy)[0] for _ in range(6)]
    # 新しいチャンクが来たら古いチャンクの残りは捨て、新しいチャンクの先頭から実行する
    assert actions == [0, 1, 100, 101, 200, 201]


def test_temporal_ensemble_weights():
    """temporal ensembling が、同じ時刻への複数の予測を重み exp(-m * i) で平均することを確かめる。

    毎ステップ推論し、その時刻をカバーする過去のチャンクの予測を全部使う。i は古い順の番号で、
    ACT 原著どおり古い予測ほど重みが大きい。チャンクの範囲を外れた予測は使わない。
    """
    m = 0.5
    policy = FakePolicy(k=3)
    ex = ChunkExecutor(chunk_size=3, temporal_ensemble=True, ensemble_coef=m)
    a0 = ex.act(policy)[0]
    a1 = ex.act(policy)[0]
    a2 = ex.act(policy)[0]
    a3 = ex.act(policy)[0]
    assert a0 == 0  # 予測が1つだけ
    w = np.exp(-m * np.arange(3))
    # t=1: 推論0 の位置1（=1）と 推論1 の位置0（=100）。古い方の重みが大きい
    assert np.isclose(a1, (w[0] * 1 + w[1] * 100) / w[:2].sum())
    # t=2: 推論0,1,2 の3つが重なる
    assert np.isclose(a2, (w[0] * 2 + w[1] * 101 + w[2] * 200) / w.sum())
    # t=3: 推論0 は時刻 0..2 しかカバーしないので外れ、推論1,2,3 の3つになる
    assert np.isclose(a3, (w[0] * 102 + w[1] * 201 + w[2] * 300) / w.sum())


def test_reset():
    """reset の後は、前のエピソードのチャンクの残りを使わずに推論し直すことを確かめる。"""
    policy = FakePolicy(k=2)
    ex = ChunkExecutor(chunk_size=2, query_every=2)
    ex.act(policy)
    ex.reset()
    assert ex.act(policy)[0] == 100  # リセット直後は必ず推論し直す
