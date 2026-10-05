"""LeRobot v3.0 形式のデータセット読み出し（借用部分: ファイルI/O層のみ）。

v3.0 形式では全エピソードが連結されて保存されている:
- data/chunk-XXX/file-XXX.parquet : 1行=1フレーム（observation.state, action, episode_index, ...）
- videos/<key>/chunk-XXX/file-XXX.mp4 : 全エピソードを連結した動画（AV1）
- meta/episodes/... : エピソードごとの範囲（dataset_from_index / to_index、動画の from_timestamp）

画像は全フレーム展開すると 20000 x 480 x 640 x 3 = 約18GB になり RAM に載せにくいので、
初回にデコードして uint8 の memmap ファイルにキャッシュする。
"""

import json
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download
from tqdm import tqdm

DATA_ROOT = Path("data")
IMAGE_KEY = "observation.images.top"


@dataclass
class Episode:
    images: np.ndarray  # (T, H, W, 3) uint8（memmap のビュー）
    state: np.ndarray  # (T, 14) float32
    action: np.ndarray  # (T, 14) float32


class AlohaDataset:
    """エピソード単位でデータを取り出すための入れ物。学習用のサンプリングは別で行う。"""

    def __init__(self, repo_id: str = "lerobot/aloha_sim_transfer_cube_human"):
        self.root = DATA_ROOT / repo_id.split("/")[-1]
        if not (self.root / "meta/info.json").exists():
            snapshot_download(repo_id, repo_type="dataset", local_dir=self.root)

        self.info = json.loads((self.root / "meta/info.json").read_text())
        self.fps = self.info["fps"]

        frames = pd.concat(
            [pd.read_parquet(f) for f in sorted(self.root.glob("data/*/*.parquet"))]
        ).sort_values("index")
        assert (frames["index"].to_numpy() == np.arange(len(frames))).all()
        self.state = np.stack(frames["observation.state"].to_numpy()).astype(np.float32)
        self.action = np.stack(frames["action"].to_numpy()).astype(np.float32)

        episodes = pd.concat(
            [pd.read_parquet(f) for f in sorted(self.root.glob("meta/episodes/*/*.parquet"))]
        ).sort_values("episode_index")
        self.ranges = list(
            zip(episodes["dataset_from_index"], episodes["dataset_to_index"], strict=True)
        )
        # 動画は1ファイルに全エピソードが連結されている前提（このデータセットでは成り立つ）。
        # その場合、動画のフレーム番号 = データセット全体の index になる。
        assert episodes[f"videos/{IMAGE_KEY}/file_index"].nunique() == 1
        video_start = np.round(episodes[f"videos/{IMAGE_KEY}/from_timestamp"] * self.fps)
        assert (video_start.to_numpy() == episodes["dataset_from_index"].to_numpy()).all()

        self.images = self._load_images()

    def __len__(self) -> int:
        return len(self.ranges)

    def __getitem__(self, i: int) -> Episode:
        start, end = self.ranges[i]
        return Episode(self.images[start:end], self.state[start:end], self.action[start:end])

    def _load_images(self) -> np.ndarray:
        h, w, c = self.info["features"][IMAGE_KEY]["shape"]
        shape = (len(self.state), h, w, c)
        cache = self.root / f"cache/{IMAGE_KEY}.npy"
        done_flag = cache.with_suffix(".done")
        if not done_flag.exists():
            cache.parent.mkdir(parents=True, exist_ok=True)
            out = np.lib.format.open_memmap(cache, mode="w+", dtype=np.uint8, shape=shape)
            video = next(self.root.glob(f"videos/{IMAGE_KEY}/*/*.mp4"))
            with av.open(str(video)) as container:
                stream = container.streams.video[0]
                stream.thread_type = "AUTO"
                n = 0
                for frame in tqdm(container.decode(stream), total=shape[0], desc="decode video"):
                    out[n] = frame.to_ndarray(format="rgb24")
                    n += 1
            assert n == shape[0], f"動画のフレーム数 {n} が parquet の行数 {shape[0]} と一致しない"
            out.flush()
            del out
            done_flag.touch()
        return np.load(cache, mmap_mode="r")
