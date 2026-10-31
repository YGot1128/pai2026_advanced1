"""LeRobot v3.0 形式のデータセット読み出し（借用部分: ファイルI/O層のみ）。

データの中身（lerobot/aloha_sim_transfer_cube_human）:
    人間が遠隔操作でキューブの受け渡しをしたデモが 50 本。1本 400 フレーム（50fps で 8 秒）、合計 20000 フレーム。
    1フレームごとに次の3つが記録されている:
        observation.images.top : 上方カメラの画像 (480, 640, 3) uint8
        observation.state      : 両腕の現在の関節角 (14,)
        action                 : 両腕の目標関節角 (14,)  … ポリシーが出力すべき値。シミュレータの env.step() にそのまま渡せる
    14次元の並びは [左腕 6関節, 左グリッパー, 右腕 6関節, 右グリッパー]（インデックス 6 と 13 がグリッパー）。
    各腕の6関節は waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate の順。

v3.0 形式では全エピソードが連結されて保存されている:
- data/chunk-XXX/file-XXX.parquet : 1行=1フレーム（observation.state, action, episode_index, index, ...）
- videos/<key>/chunk-XXX/file-XXX.mp4 : 全エピソードを連結した動画（AV1）
- meta/episodes/... : エピソードごとの範囲（dataset_from_index / to_index、動画の from_timestamp）

そこで、全フレームを「データセット全体の通し番号（index）」で並べた1本の配列として持ち、
エピソード i は ranges[i] = (start, end) の範囲 [start, end) として切り出す。

画像は全フレーム展開すると 20000 x 480 x 640 x 3 = 約18GB になり RAM に載せにくいので、
初回にデコードして uint8 の memmap ファイルにキャッシュする。
memmap はディスク上のファイルを配列のように見せる仕組みで、アクセスした部分だけが読み込まれる。
"""

import json
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download
from tqdm import tqdm

DATA_ROOT = Path("data")  # ダウンロード先とキャッシュの置き場所（.gitignore 済み）
IMAGE_KEY = "observation.images.top"


@dataclass
class Episode:
    """1本のデモ。T はフレーム数（このデータセットでは 400）。

    時刻 t の state[t] を観測した状態で action[t] を指令した、という対応になっている。
    """

    images: np.ndarray  # (T, H, W, 3) uint8（memmap のビュー。np.asarray で実体化される）
    state: np.ndarray  # (T, 14) float32: 現在の関節角
    action: np.ndarray  # (T, 14) float32: 目標関節角


class AlohaDataset:
    """エピソード単位でデータを取り出すための入れ物。学習用のサンプリングは別で行う（chunk_dataset.py）。

    初回はデータセットのダウンロード（Hugging Face Hub）と動画のデコードが走り、data/ 以下にキャッシュする。
    2回目以降はキャッシュを読むだけなので速い。

    Args:
        repo_id: Hugging Face Hub のデータセット ID。

    Attributes:
        fps: フレームレート（50）。
        state, action: (N, 14) float32。N はデータセット全体のフレーム数（20000）。
        images: (N, H, W, 3) uint8 の memmap。
        ranges: エピソードごとの [start, end) の範囲。ds.state[start:end] がエピソードの state。

    使い方:
        ds = AlohaDataset()
        len(ds)        # エピソード数 50
        ep = ds[0]     # 0 番目のエピソード（Episode）
    """

    def __init__(self, repo_id: str = "lerobot/aloha_sim_transfer_cube_human"):
        self.root = DATA_ROOT / repo_id.split("/")[-1]
        if not (self.root / "meta/info.json").exists():
            snapshot_download(repo_id, repo_type="dataset", local_dir=self.root)

        self.info = json.loads((self.root / "meta/info.json").read_text())
        self.fps = self.info["fps"]

        # 状態と行動: parquet を全部読み、通し番号 index の順に並べる
        frames = pd.concat(
            [pd.read_parquet(f) for f in sorted(self.root.glob("data/*/*.parquet"))]
        ).sort_values("index")
        assert (frames["index"].to_numpy() == np.arange(len(frames))).all()
        # parquet の1セルには長さ14の配列が入っているので、np.stack で (N, 14) の2次元配列にする
        self.state = np.stack(frames["observation.state"].to_numpy()).astype(np.float32)
        self.action = np.stack(frames["action"].to_numpy()).astype(np.float32)

        # エピソードの範囲
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
        """エピソード数。"""
        return len(self.ranges)

    def __getitem__(self, i: int) -> Episode:
        """i 番目のエピソードを返す（配列はコピーせずスライスで切り出す）。"""
        start, end = self.ranges[i]
        return Episode(self.images[start:end], self.state[start:end], self.action[start:end])

    def _load_images(self) -> np.ndarray:
        """全フレームの画像を memmap として返す。キャッシュがなければ動画をデコードして作る。

        デコードが途中で中断されると不完全なキャッシュが残るので、最後まで書き終えたときだけ
        目印のファイル（.done）を作り、それがあるかどうかでキャッシュの完成を判断する。
        """
        h, w, c = self.info["features"][IMAGE_KEY]["shape"]
        shape = (len(self.state), h, w, c)
        cache = self.root / f"cache/{IMAGE_KEY}.npy"
        done_flag = cache.with_suffix(".done")
        if not done_flag.exists():
            cache.parent.mkdir(parents=True, exist_ok=True)
            # .npy 形式のヘッダ付きで memmap を作る（np.load(mmap_mode="r") でそのまま開ける）
            out = np.lib.format.open_memmap(cache, mode="w+", dtype=np.uint8, shape=shape)
            video = next(self.root.glob(f"videos/{IMAGE_KEY}/*/*.mp4"))
            with av.open(str(video)) as container:
                stream = container.streams.video[0]
                stream.thread_type = "AUTO"  # デコードをマルチスレッドにする
                n = 0
                for frame in tqdm(container.decode(stream), total=shape[0], desc="decode video"):
                    out[n] = frame.to_ndarray(format="rgb24")
                    n += 1
            assert n == shape[0], f"動画のフレーム数 {n} が parquet の行数 {shape[0]} と一致しない"
            out.flush()
            del out
            done_flag.touch()
        return np.load(cache, mmap_mode="r")
