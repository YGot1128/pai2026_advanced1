"""(実行) 評価で保存したロールアウトの動画を横に並べ、ラベルを付けて、提出動画の素材を作る。

使い方（リポジトリ直下で実行）:
    bash scripts/make_videos.sh                       # 下の CLIPS をすべて作る
    bash scripts/make_videos.sh --only no2_heads_seed3
    bash scripts/make_videos.sh --list                # 素材の一覧と、元の動画が揃っているか
    （中身は uv run python -m pai_act.visualize.videos ...）

出力（既定は outputs/video/。outputs/ は .gitignore 済み）:
    <名前>.mp4          … 1920x1080、50 fps、h264（yuv420p）。YouTube にそのまま上げられる形式
    <名前>.sources.md   … 使った評価の出力先・動画のファイル・そのエピソードの結果（results.json の値）

並べ方の考え方:
    - 評価の動画は 1 ステップ 1 フレーム（50 fps）なので、先頭をそろえて並べれば同じ時刻 t のフレームが並ぶ。
      1 本ずつ速さを変えると t がずれるので変えない
    - 成功したエピソードは成功したステップで終わり、長さがそろわない。短いほうは最後のフレーム
      （SUCCESS / FAIL の表示）を止めたまま、長いほうが終わるまで見せる。ループさせると、
      どちらが先に終わったかが分からなくなる
    - 比較の中で共通の条件は上の 1 行に、画面ごとに違う条件だけを各画面の下の帯に書く
    - 帯の色はレポートの図と同じヘッドの色（report/figures/make_fig6.py の COLOR）
    - ラベルの文字の間隔は、レポートの図と同じ規則（数字と日本語の間に半角スペースを入れない。
      英単語の前後と、数字と英字の単位の間には入れる）
    - 評価コードが動画の左上に描き込んだ「seed 3  t=…  reward=…」は残す。ラベルの「評価シード」と
      同じものだと分かり、t は ensembling なしで推論し直す時刻（100・200・300）を見るのに使う
    - タイトル・図・説明文（「50 本中の 1 例」など）は動画編集ソフトで足す前提で、ここでは入れない
"""

import argparse
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import matplotlib_fontja

FRAME_W, FRAME_H = 1920, 1080
FPS = 50  # 評価の動画と同じ（gym-aloha の render_fps）
BG = "white"
TEXT = "0x1a1a1a"
# レポートの図と同じヘッドの色（report/REPORT_HANDOVER.md の 3 節、10/27）
COLOR = {"ACT": "0x2a78d6", "Diffusion": "0xeb6834", "Flow Matching": "0x1baf7a"}
FONT = Path(matplotlib_fontja.__file__).parent / "fonts" / "ipaexg.ttf"


@dataclass
class Panel:
    """並べる 1 画面。label の行に {ms} があれば、predict_chunk の中央値に置き換える。"""

    eval_dir: str  # runs/ からの評価の出力先
    seed: int  # 評価シード（キューブの初期位置を決める）
    head: str  # 帯の色を決める（COLOR のキー）
    label: list[str]  # 帯に書く行（1〜2 行）


@dataclass
class Clip:
    header: str | None  # 比較の中で共通の条件。None なら上の行を置かない
    panels: list[Panel]
    scale: float  # 640x480 の元の動画を何倍にするか


IN_DIST = "評価シード3（学習デモと同じ配置）"

# 動画の素材。番号は report/REPORT_HANDOVER.md の 12 節の No. と同じ。
# No.2〜4 は評価シード 3 にそろえ、同じ配置で 1 か所だけ変えたことが伝わるようにした
CLIPS: dict[str, Clip] = {
    "no1_act_seed3": Clip(
        header=None,
        panels=[Panel("act/eval_ckpt_last", 3, "ACT", [f"ACT　{IN_DIST}"])],
        scale=1.5,
    ),
    "no1_act_seed51": Clip(
        header=None,
        panels=[
            Panel("act/eval_ckpt_last_s50n150", 51, "ACT", ["ACT　評価シード51（学習外の配置）"])
        ],
        scale=1.5,
    ),
    "no2_heads_seed3": Clip(
        header=IN_DIST,
        panels=[
            Panel("act/eval_ckpt_last", 3, "ACT", ["ACT", "推論{ms}"]),
            Panel(
                "diffusion/eval_ckpt_last_ddpm100",
                3,
                "Diffusion",
                ["Diffusion", "推論{ms}（デノイズ100回）"],
            ),
            Panel(
                "flow_matching/eval_ckpt_last_euler10",
                3,
                "Flow Matching",
                ["Flow Matching", "推論{ms}（Euler 10回）"],
            ),
        ],
        scale=1.0,
    ),
    "no3_diffusion_ensembling_seed3": Clip(
        header=f"Diffusion　{IN_DIST}",
        panels=[
            Panel("diffusion/eval_ckpt_last_ddpm100", 3, "Diffusion", ["temporal ensemblingあり"]),
            Panel(
                "diffusion/eval_ckpt_last_ddpm100_noens100",
                3,
                "Diffusion",
                ["temporal ensemblingなし（100ステップごとに推論）"],
            ),
        ],
        scale=1.5,
    ),
    "no4_diffusion_20k_80k_seed3": Clip(
        header=f"Diffusion　{IN_DIST}",
        panels=[
            Panel("diffusion/eval_ckpt_020000_ddpm100", 3, "Diffusion", ["学習20kステップ"]),
            Panel("diffusion/eval_ckpt_last_ddpm100", 3, "Diffusion", ["学習80kステップ"]),
        ],
        scale=1.5,
    ),
}

HEADER_H = 110  # 上の 1 行の高さ（px）
LINE_H = {1: 70, 2: 110}  # 帯の高さ（帯の行数ごと）
FONT_HEADER, FONT_LINE1, FONT_LINE2 = 44, 36, 28


def find_video(eval_dir: Path, seed: int) -> Path:
    """評価の出力先から、そのシードの動画を 1 本探す（ファイル名は ep<i>_seed<s>_<結果>.mp4）。"""
    found = sorted((eval_dir / "videos").glob(f"ep*_seed{seed}_*.mp4"))
    if len(found) != 1:
        raise FileNotFoundError(f"{eval_dir}/videos に seed {seed} の動画が {len(found)} 本ある")
    return found[0]


def load_episode(eval_dir: Path, seed: int) -> tuple[dict, float]:
    """results.json から、そのシードのエピソードの結果と predict_chunk の中央値（ms）を返す。"""
    result = json.loads((eval_dir / "results.json").read_text())
    episode = next(e for e in result["episodes"] if e["seed"] == seed)
    return episode, result["predict_ms"]["median"]


def fmt_ms(ms: float) -> str:
    """10 ms 未満は小数 1 桁（6.5 ms）、それ以上は整数（14 ms、125 ms）。README の書き方に合わせた。"""
    return f"{ms:.1f} ms" if ms < 10 else f"{ms:.0f} ms"


def count_frames(path: Path) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames"]
        + ["-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(out.stdout.strip())


def escape(text: str) -> str:
    """drawtext の text= に渡すため、フィルタの区切りに使われる文字を逃がす。"""
    for ch in ["\\", "'", ":", ",", ";", "[", "]"]:
        text = text.replace(ch, "\\" + ch)
    return text


def drawtext(text: str, size: int, color: str, y: str) -> str:
    """横方向は中央、縦方向の位置は y（ffmpeg の式）で指定する。expansion=none で % を展開させない。"""
    return (
        f"drawtext=fontfile='{FONT}':text='{escape(text)}':expansion=none"
        f":fontsize={size}:fontcolor={color}:x=(w-text_w)/2:y={y}"
    )


def build_filter(clip: Clip, labels: list[list[str]], frames: list[int], hold: float) -> str:
    """ffmpeg の filter_complex を組み立てる。

    各画面: 長さをそろえる（最後のフレームを止める）→ 拡大 → 下に帯を足して色を塗り、ラベルを書く。
    全体: 横に並べ → 1920x1080 の中央に置き、上に共通の条件を書く → 最後に hold 秒止める。
    """
    w, h = round(640 * clip.scale), round(480 * clip.scale)
    n_lines = max(len(lab) for lab in labels)
    band = LINE_H[n_lines]
    longest = max(frames)
    parts = []
    for i, (panel, lab, n) in enumerate(zip(clip.panels, labels, frames, strict=True)):
        chain = [f"tpad=stop_mode=clone:stop={longest - n}"]
        if clip.scale != 1.0:
            chain.append(f"scale={w}:{h}:flags=lanczos")
        chain.append(f"pad=w={w}:h={h + band}:x=0:y=0:color={BG}")
        chain.append(f"drawbox=x=0:y={h}:w={w}:h={band}:color={COLOR[panel.head]}:t=fill")
        if len(lab) == 1:
            chain.append(drawtext(lab[0], FONT_LINE1, "white", f"{h}+({band}-text_h)/2"))
        else:
            chain.append(drawtext(lab[0], FONT_LINE1, "white", f"{h}+14"))
            chain.append(drawtext(lab[1], FONT_LINE2, "white", f"{h}+{band}-text_h-16"))
        parts.append(f"[{i}:v]" + ",".join(chain) + f"[p{i}]")

    n = len(clip.panels)
    if n > 1:
        parts.append("".join(f"[p{i}]" for i in range(n)) + f"hstack=inputs={n}[s]")
    else:
        parts.append("[p0]null[s]")

    header_h = HEADER_H if clip.header else 0
    content_h = header_h + h + band
    top = (FRAME_H - content_h) // 2
    final = [f"pad=w={FRAME_W}:h={FRAME_H}:x=(ow-iw)/2:y={top + header_h}:color={BG}"]
    if clip.header:
        final.append(drawtext(clip.header, FONT_HEADER, TEXT, f"{top}+({header_h}-text_h)/2"))
    final.append(f"tpad=stop_mode=clone:stop_duration={hold}")
    final.append("format=yuv420p")
    parts.append("[s]" + ",".join(final) + "[out]")
    return ";".join(parts)


def make_clip(name: str, clip: Clip, runs: Path, out_dir: Path, hold: float) -> Path:
    videos, labels, frames, rows = [], [], [], []
    for panel in clip.panels:
        eval_dir = runs / panel.eval_dir
        video = find_video(eval_dir, panel.seed)
        episode, ms = load_episode(eval_dir, panel.seed)
        videos.append(video)
        labels.append([line.replace("{ms}", fmt_ms(ms)) for line in panel.label])
        frames.append(count_frames(video))
        rows.append((panel, video, episode, ms, labels[-1], frames[-1]))

    out = out_dir / f"{name}.mp4"
    cmd = ["ffmpeg", "-y", "-v", "error"]
    for v in videos:
        cmd += ["-i", str(v)]
    cmd += ["-filter_complex", build_filter(clip, labels, frames, hold), "-map", "[out]"]
    cmd += ["-r", str(FPS), "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-an", str(out)]
    subprocess.run(cmd, check=True)
    write_sources(name, clip, rows, hold, out_dir)
    return out


def write_sources(name: str, clip: Clip, rows: list, hold: float, out_dir: Path) -> None:
    """どの評価のどの動画から作ったかを残す（レポートの図の fig6_sources.md と同じ目的）。"""
    lines = [
        f"# {name}.mp4 の出典",
        "",
        "`bash scripts/make_videos.sh` が書き出したもの（src/pai_act/visualize/videos.py）。",
        "",
        f"- 上の行（共通の条件）: {clip.header or '（なし）'}",
        f"- 拡大: {clip.scale} 倍、最後に {hold} 秒止める。短い画面は最後のフレームを止めて長さをそろえた",
        "",
        "| 画面 | 帯のラベル | 動画 | フレーム数 | 成否 | 最大報酬 | ステップ数 | 失敗の型 "
        "| predict_chunk の中央値 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for i, (_panel, video, ep, ms, label, n) in enumerate(rows, start=1):
        ok = "成功" if ep["success"] else "失敗"
        lines.append(
            f"| {i} | {' / '.join(label)} | `{video}` | {n} | {ok} | {ep['max_reward']} "
            f"| {ep['steps']} | {ep['failure_type'] or '—'} | {ms:.1f} ms |"
        )
    lines += [
        "",
        "結果は、各動画と同じ評価の出力先（動画の 2 つ上のディレクトリ）の results.json の値。",
    ]
    (out_dir / f"{name}.sources.md").write_text("\n".join(lines) + "\n")


def list_clips(runs: Path) -> None:
    for name, clip in CLIPS.items():
        status = []
        for p in clip.panels:
            try:
                status.append(find_video(runs / p.eval_dir, p.seed).name)
            except FileNotFoundError as e:
                status.append(f"（ない: {e}）")
        print(f"{name}: " + ", ".join(status))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--only", nargs="+", choices=list(CLIPS), help="作る素材（既定はすべて）")
    parser.add_argument("--list", action="store_true", help="素材の一覧と、元の動画があるか")
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/video"))
    parser.add_argument("--hold", type=float, default=1.0, help="最後のフレームを止める秒数")
    args = parser.parse_args()

    if args.list:
        list_clips(args.runs)
        return
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg が見つからない（drawtext が使える ffmpeg が要る）")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name in args.only or list(CLIPS):
        out = make_clip(name, CLIPS[name], args.runs, args.out_dir, args.hold)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
