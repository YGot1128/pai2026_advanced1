"""ダウンロード済みデータセット（LeRobot v3.0 形式）の中身を確認する。

dataset.py の AlohaDataset は動画を全フレームデコードしてキャッシュを作るが、
ここではキャッシュは作らず、元のファイル（parquet / mp4 / json）を直接読んで中身と整合性を表示する。

確認する項目（上から順に実行し、どこかで失敗したらそこで止まる）:
    1. check_files:    どのファイルがあるか（パスとサイズ）
    2. check_info:     meta/info.json の概要（fps・エピソード数・各特徴量の形）
    3. check_episodes: meta/episodes の各エピソードの範囲（フレーム番号と動画内の時刻）
    4. check_frames:   data/ の parquet（1行=1フレーム）がエピソードの範囲と一致するか
    5. check_video:    mp4 のフレーム数・長さと、指定エピソードの先頭フレームを画像として保存

使い方（リポジトリ直下で実行）:
    uv run python -m pai_act.data.check_dataset                 # エピソード 0 の先頭フレームを保存
    uv run python -m pai_act.data.check_dataset --episodes 0 49  # 複数指定も可
出力: outputs/check_dataset/epXX_first.png
"""

import argparse
import json
from pathlib import Path

import av
import imageio
import numpy as np
import pandas as pd

from pai_act.data.dataset import DATA_ROOT, IMAGE_KEY


def read_parquets(root: Path, pattern: str) -> pd.DataFrame:
    """root 以下で pattern に一致する parquet をすべて読み、1つの表につなげる。"""
    files = sorted(root.glob(pattern))
    assert files, f"{root / pattern} が見つからない"
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def check_files(root: Path) -> None:
    """データセットのファイル一覧をサイズ付きで表示する（.cache/ と cache/ は除く）。"""
    print(f"root: {root}")
    for f in sorted(root.rglob("*")):
        rel = f.relative_to(root)
        if f.is_file() and rel.parts[0] not in (".cache", "cache"):
            print(f"  {rel}  ({f.stat().st_size / 1e6:.1f} MB)")


def check_info(root: Path) -> dict:
    """meta/info.json の概要を表示して、その中身を返す。"""
    info = json.loads((root / "meta/info.json").read_text())
    print(f"format: {info['codebase_version']}, robot: {info['robot_type']}, fps: {info['fps']}")
    print(f"episodes: {info['total_episodes']}, frames: {info['total_frames']}")
    for key, feat in info["features"].items():
        print(f"  {key:<24} {feat['dtype']:<8} {feat['shape']}")
    tasks = pd.read_parquet(root / "meta/tasks.parquet")
    for task, row in tasks.iterrows():
        print(f"  task {row['task_index']}: {task}")
    return info


def check_episodes(root: Path, info: dict) -> pd.DataFrame:
    """meta/episodes の範囲を表示し、フレーム番号と動画の時刻が対応しているかを確かめる。

    各エピソードについて
        dataset_from_index / dataset_to_index : データセット全体の通し番号での範囲 [from, to)
        from_timestamp / to_timestamp         : 動画ファイルの中での開始・終了時刻（秒）
    が記録されている。動画が1本に連結されていれば、時刻 × fps がそのまま通し番号になる。
    """
    episodes = read_parquets(root, "meta/episodes/*/*.parquet").sort_values("episode_index")
    video = f"videos/{IMAGE_KEY}"
    table = pd.DataFrame(
        {
            "episode": episodes["episode_index"],
            "length": episodes["length"],
            "from_index": episodes["dataset_from_index"],
            "to_index": episodes["dataset_to_index"],
            "video_file": episodes[f"{video}/chunk_index"].map("chunk-{:03d}".format)
            + "/"
            + episodes[f"{video}/file_index"].map("file-{:03d}.mp4".format),
            "from_ts": episodes[f"{video}/from_timestamp"],
            "to_ts": episodes[f"{video}/to_timestamp"],
        }
    )
    print(table.to_string(index=False, max_rows=12))

    assert len(table) == info["total_episodes"], "エピソード数が info.json と一致しない"
    assert (table["to_index"] - table["from_index"] == table["length"]).all()
    # 前のエピソードの終わりと次のエピソードの始まりが隙間なくつながっているか
    assert (table["from_index"].iloc[1:].to_numpy() == table["to_index"].iloc[:-1].to_numpy()).all()
    assert table["to_index"].iloc[-1] == info["total_frames"]
    fps = info["fps"]
    assert np.allclose(table["from_ts"] * fps, table["from_index"]), (
        "動画の時刻と通し番号がずれている"
    )
    assert np.allclose((table["to_ts"] - table["from_ts"]) * fps, table["length"])
    print(f"OK: {len(table)} episodes, lengths {table['length'].min()}-{table['length'].max()}")
    return episodes


def check_frames(root: Path, info: dict, episodes: pd.DataFrame) -> None:
    """data/ の parquet を読み、各エピソードの行が meta/episodes の範囲と一致するかを確かめる。"""
    files = sorted(root.glob("data/*/*.parquet"))
    frames = read_parquets(root, "data/*/*.parquet").sort_values("index")
    print(f"{len(files)} parquet files, {len(frames)} rows, columns: {list(frames.columns)}")
    assert len(frames) == info["total_frames"], "フレーム数が info.json と一致しない"
    assert (frames["index"].to_numpy() == np.arange(len(frames))).all(), "通し番号が欠けている"

    for _, ep in episodes.iterrows():
        rows = frames.iloc[ep["dataset_from_index"] : ep["dataset_to_index"]]
        assert (rows["episode_index"] == ep["episode_index"]).all()
        assert (rows["frame_index"].to_numpy() == np.arange(ep["length"])).all()

    # parquet の1セルには長さ14の配列が入っているので、np.stack で (N, 14) にする
    for key in ("observation.state", "action"):
        x = np.stack(frames[key].to_numpy())
        print(f"  {key:<18} shape {x.shape}, min {x.min():+.3f}, max {x.max():+.3f}")
    print("OK: rows match episode ranges")


def check_video(root: Path, info: dict, episodes: pd.DataFrame, show: list[int], out_dir: Path):
    """mp4 のストリーム情報を表示し、show で指定したエピソードの先頭フレームを PNG に保存する。

    動画は全エピソードの連結なので、エピソード i の映像は from_timestamp の位置から始まる。
    シークはキーフレーム単位でしか飛べないため、手前のキーフレームから目的の時刻までデコードを進める。
    """
    files = sorted(root.glob(f"videos/{IMAGE_KEY}/*/*.mp4"))
    print(f"{len(files)} video file(s)")
    out_dir.mkdir(parents=True, exist_ok=True)
    episodes = episodes.set_index("episode_index")
    for video in files:
        with av.open(str(video)) as container:
            stream = container.streams.video[0]
            assert stream.duration is not None, f"{video}: 長さの情報がない"
            duration = float(stream.duration * stream.time_base)
            print(
                f"  {video.relative_to(root)}: {stream.codec_context.name}, "
                f"{stream.width}x{stream.height}, {stream.frames} frames, {duration:.1f} s"
            )
            for i in show:
                ts = episodes.loc[i, f"videos/{IMAGE_KEY}/from_timestamp"]
                container.seek(int(ts / stream.time_base), stream=stream)
                # 浮動小数の誤差で目的のフレームを飛ばさないよう、半フレーム分の余裕を持たせる
                frame = next(
                    f for f in container.decode(stream) if f.time >= ts - 0.5 / info["fps"]
                )
                path = out_dir / f"ep{i:02d}_first.png"
                imageio.imwrite(path, frame.to_ndarray(format="rgb24"))
                print(f"  episode {i}: t={frame.time:.2f} s -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default="aloha_sim_transfer_cube_human", help="data/ 以下の名前")
    parser.add_argument(
        "--episodes", type=int, nargs="+", default=[0], help="先頭フレームを保存するエピソード"
    )
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/check_dataset"))
    args = parser.parse_args()

    root = DATA_ROOT / args.name
    assert (root / "meta/info.json").exists(), (
        f"{root} にデータがない（先に AlohaDataset() を一度作ってダウンロードする）"
    )

    print("== files ==")
    check_files(root)
    print("\n== info ==")
    info = check_info(root)
    print("\n== episodes ==")
    episodes = check_episodes(root, info)
    print("\n== frames ==")
    check_frames(root, info, episodes)
    print("\n== video ==")
    check_video(root, info, episodes, args.episodes, args.out_dir)
    print("\nALL OK")


if __name__ == "__main__":
    main()
