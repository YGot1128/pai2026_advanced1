"""学習サンプルの切り出し: 観測1フレーム + その時刻から先の行動 k ステップ（行動チャンク）。

ACT の学習では「時刻 t の観測を見て、t から t+k-1 までの k ステップ分の行動をまとめて予測する」。
1本のデモ（T フレーム）から、t = 0, 1, ..., T-1 のすべてを起点にサンプルを作る。

    観測:   image[t], state[t]
    目標:   action[t], action[t+1], ..., action[t+k-1]

エピソード終端付近では t+k-1 がエピソードの外にはみ出す。はみ出した分は
最後の行動を繰り返して埋め（padding）、`action_is_pad` で「ここは埋め草」と印を付ける。
損失はこの印が付いていないステップだけで計算する（policies/base.py の masked_l1）。
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class ChunkDataset(Dataset):
    """エピソード群から (観測, 行動チャンク) のサンプルを取り出す PyTorch Dataset。

    Args:
        source: エピソードの入れ物。`images (N, H, W, 3) uint8`, `state (N, 14)`, `action (N, 14)`,
            `ranges [(start, end), ...]` を持つもの（通常は data.dataset.AlohaDataset）。
            N はデータセット全体のフレーム数で、エピソード i は [start, end) の範囲。
        episodes: 使うエピソード番号のリスト（学習用 / 検証用の分割に使う）。
        chunk_size: 行動チャンクの長さ k。

    1サンプルは次のキーを持つ dict（DataLoader でまとめると先頭にバッチ次元 B が付く）:
        image:         (3, H, W) uint8   … 0〜255 のまま返し、float 化と正規化はモデル側で行う
        state:         (14,) float32     … 時刻 t の関節角
        action:        (k, 14) float32   … 時刻 t から k ステップ分の目標関節角
        action_is_pad: (k,) bool         … True のステップはエピソード外（損失に含めない）

    Note:
        source.images は約18GB の memmap（ディスク上のファイルを配列として見せるもの）。
        DataLoader の num_workers > 0 では、Linux の既定（fork）でワーカーを作るため配列はコピーされず、
        各ワーカーが同じファイルを読む。spawn 方式だと memmap が pickle されて全データがコピーされるので注意。
    """

    def __init__(self, source, episodes: list[int], chunk_size: int):
        self.source = source
        self.chunk_size = chunk_size
        # サンプル番号 → (データセット全体でのフレーム番号 t, そのエピソードの終端 end)
        self.index: list[tuple[int, int]] = []
        for ep in episodes:
            start, end = source.ranges[ep]
            self.index.extend((t, end) for t in range(start, end))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        t, end = self.index[index]
        k = self.chunk_size

        # [t, t+k) のうちエピソード内にある部分を取り、足りない分は最後の行動で埋める
        action = self.source.action[t : min(t + k, end)]
        n_valid = len(action)
        if n_valid < k:
            action = np.concatenate([action, np.repeat(action[-1:], k - n_valid, axis=0)])
        is_pad = np.arange(k) >= n_valid

        # memmap は読み取り専用なので、np.array で1フレーム分だけメモリにコピーしてから Tensor にする。
        # (H, W, 3) → (3, H, W): PyTorch の畳み込みはチャンネルが先頭の形を期待する
        image = torch.from_numpy(np.array(self.source.images[t])).permute(2, 0, 1)
        return {
            "image": image,
            "state": torch.from_numpy(np.asarray(self.source.state[t])),
            "action": torch.from_numpy(np.ascontiguousarray(action)),
            "action_is_pad": torch.from_numpy(is_pad),
        }
