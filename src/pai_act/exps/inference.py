"""推論時の行動チャンクの実行: action chunking と temporal ensembling。

ポリシーは1回の推論で k ステップ分の行動（チャンク）を出すが、シミュレータには1ステップずつ行動を渡す。
その橋渡しのやり方が2通りある（ACT 原著 Sec. III-C）。

1. action chunking（query_every = n）
   n ステップに1回だけ推論し、出てきたチャンクの先頭 n 個を順に実行する。
   推論回数が減り、1回の予測で「一連の動き」を出すので、人間のデモの細かい停止や揺れに引きずられにくい。
   ただし n ステップの間は観測を見ないので、途中で状況が変わっても反応できない。
   また、チャンクの切れ目で動きが不連続になりやすい。

2. temporal ensembling（毎ステップ推論）
   毎ステップ推論すると、時刻 t の行動について「過去 k 回分の推論がそれぞれ予測した値」が手元に揃う。
   それらを重み w_i = exp(-m * i) で平均する（i = 0 が最も古い予測）。
   古い予測ほど重みが大きく、新しい予測は少しずつ反映される → 動きが滑らかになる。
   m を大きくするほど最も古い予測に寄り、小さくするほど単純平均に近づく。

        時刻:        t-2   t-1    t    t+1   t+2
        2つ前の推論: [a0    a1    a2    a3   ...]
        1つ前の推論:       [a0    a1    a2   ...]
        今回の推論:              [a0    a1   ...]
                                  ↑ 時刻 t ではこの3つ（縦に並んだ値）を重み付き平均する
"""

from collections import deque
from collections.abc import Callable

import numpy as np


class ChunkExecutor:
    """行動チャンクを1ステップずつの行動に変換する（エピソードごとに reset() する）。

    Args:
        chunk_size: チャンクの長さ k。
        query_every: 何ステップごとに推論するか。temporal_ensemble=True のときは無視（毎ステップ）。
        temporal_ensemble: temporal ensembling を使うか。
        ensemble_coef: 重み exp(-m * i) の m。

    使い方:
        executor.reset()
        for t in range(T):
            action = executor.act(lambda: policy で観測から (k, 14) のチャンクを予測する関数)
    """

    def __init__(
        self,
        chunk_size: int,
        query_every: int = 1,
        temporal_ensemble: bool = False,
        ensemble_coef: float = 0.01,
    ):
        if temporal_ensemble:
            query_every = 1
        assert 1 <= query_every <= chunk_size
        self.chunk_size = chunk_size
        self.query_every = query_every
        self.temporal_ensemble = temporal_ensemble
        self.ensemble_coef = ensemble_coef
        self.reset()

    def reset(self) -> None:
        self.t = 0
        # (推論した時刻, チャンク (k, D)) を古い順に保持する。時刻 t をカバーしなくなったものは捨てる
        self.chunks: deque[tuple[int, np.ndarray]] = deque()

    def act(self, predict_chunk: Callable[[], np.ndarray]) -> np.ndarray:
        """現在の時刻の行動を返し、時刻を1つ進める。

        Args:
            predict_chunk: 引数なしで (k, D) のチャンクを返す関数。推論が必要な時刻にだけ呼ばれる
                （呼ぶ側で現在の観測を閉じ込めておく）。
        """
        if self.t % self.query_every == 0:
            self.chunks.append((self.t, predict_chunk()))
        # 時刻 t をカバーしているのは start <= t < start + k のチャンクだけ
        while self.chunks and self.chunks[0][0] + self.chunk_size <= self.t:
            self.chunks.popleft()

        if self.temporal_ensemble:
            preds = np.stack([chunk[self.t - start] for start, chunk in self.chunks])  # 古い順
            weights = np.exp(-self.ensemble_coef * np.arange(len(preds)))
            action = (weights[:, None] * preds).sum(axis=0) / weights.sum()
        else:
            start, chunk = self.chunks[-1]
            action = chunk[self.t - start]

        self.t += 1
        return action
