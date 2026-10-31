"""(実行) すべての評価結果（runs/*/eval_*/results.json）を1つの表（CSV）にまとめる。

使い方（リポジトリ直下で実行）:
    bash scripts/summarize.sh                  # outputs/summary/ に runs.csv と episodes_long.csv を作る
    bash scripts/summarize.sh --include-test   # 動作確認用の *_test の学習も含める
    （中身は uv run python -m pai_act.visualize.summary ...）

compare.py との違い: compare.py は決めておいた条件の組（プリセット）を比べる表と図を作る。
こちらは runs/ にある評価を全部自動で集め、条件・成功率・推論時間・エピソードごとの結果を横に並べる。
新しい条件を学習・評価すれば、コードを変えずに次の実行から行が増える。

出力（outputs/summary/。outputs/ は .gitignore 済み）:
    runs.csv           … 1 行 = 1 回の評価。左から 条件 → 指標 → エピソードごとの段階（ep00〜ep49）
    episodes_long.csv  … 1 行 = 1 エピソード（条件 × シード）。Excel のピボットテーブル用
    extended/runs.csv, extended/episodes_long.csv
                       … 評価の本数を増やした条件だけの表（列の形は上と同じ）。
                         評価のシードの範囲を既定（シード 0 から 50 本）から変えた評価（eval_*_s50n150 など）は、
                         上の表には入れず、同じ条件の既定の 50 本と合わせて 1 行にしてこちらに書く
                         （例: シード 0〜49 と 50〜199 を合わせた 200 本）。本数の違う行が同じ表に混ざると、
                         成功率の精度（信頼区間の幅）が行ごとに違うのに見分けにくいため。
                         合わせた 200 本の指標に加えて、学習内の配置（シード 0〜49、in_*）と
                         学習外の配置（シード 50 以降、ood_*）に分けた成功率・信頼区間・失敗の内訳も出す。
                         シード 0〜49 のキューブの配置は学習デモと同じなので、合わせた値だけでは
                         汎化を論じられないため（docs/EVALUATION.md）

Excel で文字化けしないよう、UTF-8 の BOM 付きで書く。再実行すると上書きするので、
Excel で書き込んだメモは別のファイル・シートに残す。

エピソードの段階（stage）: エピソード中の最大報酬 max_reward（0〜4）をそのまま使う。大きいほど成功に近い。
    0: no_touch（触れなかった） 1: not_lifted（触れたが持ち上げられなかった）
    2: not_transferred（持ち上げたが左が触れなかった） 3: transfer_failed（左が触れたが持ち上げられなかった）
    4: success
    定義は docs/EVALUATION.md。ep 列の番号は評価のシード（runs.csv はどの条件もシード 0〜49 を使うので、
    同じ列は同じキューブの初期位置になる）。episodes_long.csv の placement 列は、そのシードの配置が
    学習内（in: シード 0〜49）か学習外（ood: シード 50 以降）か。Excel の条件付き書式（カラースケール）で塗ると、
    失敗しやすいシード（縦に並んだ低い値）と、段階が成功に近づいた条件（行の値の増加）が見える。

条件の列:
    学習の設定（runs/<name>/config.json）は、表に入る評価の間で値が違う項目だけを列にする。
    全条件で同じ値の項目（chunk_size など）は列にせず、実行時に「共通の設定」として表示する。
    ある手法が使わない設定（ACT の num_diffusion_steps など）は空欄にする（MODEL_FIELDS）。
    推論の設定（サンプラー・ステップ数・クリップ・η）は、学習時の設定ではなく results.json に残った
    実際に使った値を使う（評価時に上書きしているため）。
"""

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np

from pai_act.visualize.compare import OUTCOMES, STAGES, wilson_interval

# 表の行の並び順（手法の順）。ここにない手法は最後に名前順で並ぶ
POLICY_ORDER = ["baseline", "act", "diffusion", "flow_matching"]

# 手法ごとに、モデルの作成に使う設定（exps/checkpoint.py の build_policy に渡しているもの）。
# ここにない設定は、その手法の行では空欄にする（使っていない値が並んで、条件の違いに見えないように）。
# 新しい手法を足したときは、build_policy と合わせてここにも足す（足さなければ全項目をそのまま出す）
MODEL_FIELDS = {
    "baseline": {"chunk_size", "hidden_dim"},
    "act": {
        "chunk_size",
        "d_model",
        "num_heads",
        "dim_feedforward",
        "num_encoder_layers",
        "num_decoder_layers",
        "num_vae_encoder_layers",
        "latent_dim",
        "dropout",
        "kl_weight",
        "learned_query_pos",
    },
    "diffusion": {
        "chunk_size",
        "d_model",
        "num_heads",
        "dim_feedforward",
        "num_encoder_layers",
        "num_decoder_layers",
        "dropout",
        "num_diffusion_steps",
    },
    "flow_matching": {
        "chunk_size",
        "d_model",
        "num_heads",
        "dim_feedforward",
        "num_encoder_layers",
        "num_decoder_layers",
        "dropout",
    },
}
# モデルの設定のうち、全手法でまとめて扱うもの（build_policy に渡さない学習の設定）は MODEL_FIELDS の対象外
ALL_MODEL_FIELDS = set().union(*MODEL_FIELDS.values())
# 学習の設定のうち、結果に関係しないもの（ログ・保存の頻度など）。列にしない
IGNORED_TRAIN_FIELDS = {
    "name",
    "resume",
    "wandb_project",
    "wandb_mode",
    "num_workers",
    "log_every",
    "print_every",
    "val_every",
    "save_every",
    "num_val_episodes",
    "val_stride",
}
# 既定の評価の範囲（exps/config.py の EvalConfig の既定値）。これ以外の範囲の評価は extended/ の表に回す
DEFAULT_EVAL_SEED = 0
DEFAULT_NUM_EPISODES = 50
# 評価の範囲を変えたときに exps/evaluate.py が出力先の名前に付ける印（例: eval_ckpt_last_s50n150）
EXTRA_SUFFIX = re.compile(r"_s\d+n\d+$")
# 評価のシード 0〜49 のキューブの配置は、学習デモ 50 本の配置と 1 対 1 で同じ（docs/EVALUATION.md）。
# それ以外のシードは学習に使っていない配置
IN_DIST_SEEDS = range(DEFAULT_EVAL_SEED, DEFAULT_EVAL_SEED + DEFAULT_NUM_EPISODES)
# 学習内 / 学習外に分けて出す指標（metrics の項目の一部）。到達段階や推論時間は合わせた値で足りるので出さない
SPLIT_METRICS = ["episodes", "successes", "success_rate", "ci95_low", "ci95_high", *OUTCOMES[:-1]]
# 推論の設定。学習時の値ではなく、results.json に残った実際に使った値を推論の列に出す
INFERENCE_FIELDS = {"sampler", "solver", "num_inference_steps", "x0_clip", "ddim_eta"}
# 値が全条件で同じでも、条件を見分けるのに使うので必ず出す学習の列
ALWAYS_TRAIN_COLUMNS = ["policy", "num_decoder_layers", "learned_query_pos", "seed", "steps"]
# 学習の列の並び順。ここにない項目（今後足した設定など）は、この後ろに名前順で並ぶ
TRAIN_COLUMN_ORDER = [
    "policy",
    "task",
    "chunk_size",
    "hidden_dim",
    "d_model",
    "num_heads",
    "dim_feedforward",
    "num_encoder_layers",
    "num_decoder_layers",
    "num_vae_encoder_layers",
    "latent_dim",
    "dropout",
    "kl_weight",
    "learned_query_pos",
    "num_diffusion_steps",
    "steps",
    "batch_size",
    "lr",
    "lr_backbone",
    "weight_decay",
    "grad_clip",
    "amp",
    "seed",
]


def natural_key(text: str) -> list:
    """数字を数として比べる並べ替えの鍵（ddim5 → ddim10 → ddim100 の順にする）。"""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", text)]


def train_conditions(config: dict) -> dict:
    """学習の設定から、列の候補（結果に関係しうる項目）を取り出す。手法が使わない項目は None にする。"""
    used = MODEL_FIELDS.get(config["policy"])
    out = {}
    for key, value in config.items():
        if key in IGNORED_TRAIN_FIELDS or key in INFERENCE_FIELDS:
            continue
        if used is not None and key in ALL_MODEL_FIELDS and key not in used:
            value = None
        out[key] = value
    return out


def eval_conditions(data: dict, train_cfg: dict, chunk_size: int, seeds: list[int]) -> dict:
    """results.json から、評価の条件（実行の仕方と、実際に使った推論の設定）を取り出す。

    サンプラーの列は Diffusion のサンプラー（ddpm / ddim）と Flow Matching の ODE ソルバー（euler / heun）を
    まとめる（どちらも「チャンクをどう生成したか」を表す）。
    seeds は行に入る全エピソードの評価のシード（複数の評価を合わせた行では、合わせた後のもの）。
    """
    cfg = data["config"]
    ens = cfg["temporal_ensemble"] == "True"
    # temporal ensembling ありでは毎ステップ推論する（exps/inference.py の ChunkExecutor）。
    # なしでは query_every ステップごと（0 はチャンク長 k）
    query_every = 1 if ens else (int(cfg["query_every"]) or chunk_size)
    sampler = data.get("sampler", {})
    name = sampler.get("sampler") or sampler.get("solver")
    steps = sampler.get("num_inference_steps")
    if name == "ddpm":
        steps = train_cfg.get("num_diffusion_steps")  # DDPM は常に T ステップ全部を辿る
    # η は記録を始める前の評価には残っていない。それまでの DDIM は η = 0（決定的）だった
    eta = sampler.get("ddim_eta", 0.0) if name == "ddim" else None
    return {
        "ckpt": Path(cfg["ckpt"]).stem,
        "eval_seeds": f"{min(seeds)}-{max(seeds)}",
        "max_steps": int(cfg["max_steps"]),
        "temporal_ensemble": ens,
        "ensemble_coef": float(cfg["ensemble_coef"]) if ens else None,
        "query_every": query_every,
        "sampler": name,
        "inference_steps": steps,
        "x0_clip": sampler.get("x0_clip"),
        "ddim_eta": eta,
    }


def metrics(episodes: list[dict], predict: dict | None) -> dict:
    """成功率・到達段階・失敗の内訳・推論時間。古い results.json（failure_types などがない）でも
    エピソードごとの max_reward から計算し直して揃える。"""
    stages = np.array([e["max_reward"] for e in episodes])
    steps = np.array([e["steps"] for e in episodes])
    n, k = len(stages), int(np.sum(stages == 4))
    lo, hi = wilson_interval(k, n)
    return {
        "episodes": n,
        "successes": k,
        "success_rate": round(k / n, 4),
        "ci95_low": round(lo, 4),
        "ci95_high": round(hi, 4),
        # 段階の平均（0〜4）。成功率が同じでも、失敗が成功に近い段階で止まっていれば大きくなる
        "mean_stage": round(float(stages.mean()), 3),
        **{
            f"reached_{s.replace(' ', '_')}": round(float(np.mean(stages >= r)), 4)
            for r, s in enumerate(STAGES, start=1)
        },
        **{name: int(np.sum(stages == r)) for r, name in enumerate(OUTCOMES)},
        "success_steps_median": float(np.median(steps[stages == 4])) if k else None,
        "predict_ms_median": round(predict["median"], 2) if predict else None,
        "predict_ms_mean": round(predict["mean"], 2) if predict else None,
    }


def split_metrics(episodes: list[dict]) -> dict:
    """学習内（in_*）と学習外（ood_*）の配置に分けた指標。学習外のシードを含まない行では空にする。

    既定の 50 本だけの表（runs.csv）は全部が学習内なので列を作らない（合わせた指標と同じ値が並ぶだけになる）。
    """
    groups = {
        "in": [e for e in episodes if e["seed"] in IN_DIST_SEEDS],
        "ood": [e for e in episodes if e["seed"] not in IN_DIST_SEEDS],
    }
    if not groups["ood"]:
        return {}
    out = {}
    for prefix, group in groups.items():
        m = metrics(group, None) if group else {}
        out.update({f"{prefix}_{k}": m.get(k) for k in SPLIT_METRICS})
    return out


def make_row(run: str, eval_name: str, train_cfg: dict, parts: list[tuple[dict, float]]) -> dict:
    """1 つの条件の評価（parts: results.json の中身と更新時刻の組。本数を増やした条件は複数）を 1 行にする。"""
    episodes = [e for data, _ in parts for e in data["episodes"]]
    seeds = [e["seed"] for e in episodes]
    # 同じシードが2回入ると、そのシードを2本と数えて本数と成功率が水増しになる
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"{run}/{eval_name}: 合わせた評価の間でシードが重なっている")
    # 推論時間は、呼んだ回数が一番多い評価のものを使う（エピソードごとの時間は残していないため。
    # 評価の間でほぼ同じ値になる）
    timed = [data["predict_ms"] for data, _ in parts if data.get("predict_ms")]
    predict = max(timed, key=lambda p: p["calls"]) if timed else None
    return {
        "condition": f"{run}/{eval_name}",
        "run": run,
        "eval": eval_name,
        # 評価した日時（results.json の更新日時。合わせた行では一番新しいもの）。条件を試した順に並べ替えるため
        "evaluated_at": datetime.fromtimestamp(max(t for _, t in parts)).strftime("%Y-%m-%d %H:%M"),
        "train": train_conditions(train_cfg),
        # 評価の条件は、シードの範囲以外はどの評価でも同じなので、最後の評価から取る
        "eval_cond": eval_conditions(parts[-1][0], train_cfg, train_cfg["chunk_size"], seeds),
        "metrics": metrics(episodes, predict),
        "split": split_metrics(episodes),
        "stages": {e["seed"]: e["max_reward"] for e in episodes},
        "steps": {e["seed"]: e["steps"] for e in episodes},
    }


def sort_rows(rows: list[dict]) -> list[dict]:
    """手法の順（POLICY_ORDER）→ 学習の名前 → 評価の名前の順に並べる。"""
    order = {p: i for i, p in enumerate(POLICY_ORDER)}
    return sorted(
        rows,
        key=lambda r: (
            order.get(r["train"]["policy"], len(order)),
            r["train"]["policy"],
            natural_key(r["run"]),
            natural_key(r["eval"]),
        ),
    )


def collect(runs_dir: Path, include_test: bool) -> tuple[list[dict], list[dict]]:
    """runs/*/eval_*/results.json を全部読み、（既定の評価の行, 本数を増やした条件の行）を返す。

    既定の評価（シード 0 から 50 本）は 1 評価 = 1 行。範囲を変えた評価（eval_ckpt_last_s50n150 など）は、
    名前から印（_s50n150）を除いた同じ条件の既定の評価と合わせて 1 行にする。
    """
    base: dict[
        tuple[str, str], tuple[dict, dict, float]
    ] = {}  # (学習, 評価) → (学習の設定, 結果, 時刻)
    extra: list[tuple[str, str, dict, dict, float]] = []
    for path in sorted(runs_dir.glob("*/eval_*/results.json")):
        eval_dir, run_dir = path.parent, path.parent.parent
        if run_dir.name.endswith("_test") and not include_test:
            continue
        config_path = run_dir / "config.json"
        if not config_path.exists():
            print(f"skip: {eval_dir}（学習の設定 {config_path} がない）")
            continue
        train_cfg = json.loads(config_path.read_text())
        data = json.loads(path.read_text())
        eval_name = eval_dir.name.removeprefix("eval_")
        mtime = path.stat().st_mtime
        is_default = (
            int(data["config"]["seed"]) == DEFAULT_EVAL_SEED
            and len(data["episodes"]) == DEFAULT_NUM_EPISODES
        )
        if is_default:
            base[(run_dir.name, eval_name)] = (train_cfg, data, mtime)
        else:
            extra.append((run_dir.name, eval_name, train_cfg, data, mtime))

    rows = [
        make_row(run, eval_name, train_cfg, [(data, mtime)])
        for (run, eval_name), (train_cfg, data, mtime) in base.items()
    ]
    extended = []
    for run, eval_name, train_cfg, data, mtime in extra:
        base_name = EXTRA_SUFFIX.sub("", eval_name)
        parts = [(data, mtime)]
        if (run, base_name) in base:
            _, base_data, base_mtime = base[(run, base_name)]
            parts.insert(0, (base_data, base_mtime))  # シードの小さい既定の評価を先に並べる
        else:
            print(
                f"注意: {run}/{eval_name} と合わせる既定の評価 eval_{base_name} がないので、単独で行にする"
            )
        extended.append(make_row(run, base_name, train_cfg, parts))
    return sort_rows(rows), sort_rows(extended)


def varying_columns(rows: list[dict], group: str, always: list[str]) -> tuple[list[str], dict]:
    """rows[*][group] の項目のうち、値が条件の間で違うもの（と always）を列にする。

    空欄（その手法が使わない設定）は比べない。全条件で同じ値の項目は (列, 共通の値) の共通の値に入れる。
    """
    keys = {k for r in rows for k in r[group]}
    columns, common = [], {}
    for key in keys:
        values = {json.dumps(r[group].get(key)) for r in rows if r[group].get(key) is not None}
        if key in always or len(values) > 1:
            columns.append(key)
        elif values:
            common[key] = json.loads(values.pop())
    rank = {k: i for i, k in enumerate(TRAIN_COLUMN_ORDER)}
    columns.sort(key=lambda k: (rank.get(k, len(rank)), k))
    return columns, common


def fmt(value) -> object:
    """CSV に書く値。None は空欄、bool は Excel が論理値として読める TRUE / FALSE にする。"""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    return value


def write_csvs(rows: list[dict], out_dir: Path) -> None:
    """runs.csv（横長: 1 行 = 1 評価）と episodes_long.csv（縦長: 1 行 = 1 エピソード）を書く。"""
    train_cols, train_common = varying_columns(rows, "train", ALWAYS_TRAIN_COLUMNS)
    eval_cols, eval_common = varying_columns(
        rows, "eval_cond", always=["temporal_ensemble", "sampler", "inference_steps"]
    )
    # 評価の列は eval_conditions に書いた順を保つ（varying_columns は学習の列の順で並べるため）
    eval_order = list(rows[0]["eval_cond"])
    eval_cols.sort(key=eval_order.index)
    seeds = sorted({s for r in rows for s in r["stages"]})

    # 評価・エピソードの列と見分けられるよう、学習のシードとステップ数は train_ を付けて呼ぶ
    def train_header(key: str) -> str:
        return f"train_{key}" if key in ("seed", "steps") else key

    def condition_cells(r: dict) -> dict:
        return {
            "condition": r["condition"],
            "run": r["run"],
            "eval": r["eval"],
            "evaluated_at": r["evaluated_at"],
            **{train_header(k): fmt(r["train"].get(k)) for k in train_cols},
            **{k: fmt(r["eval_cond"].get(k)) for k in eval_cols},
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    wide = [
        {
            **condition_cells(r),
            **{k: fmt(v) for k, v in r["metrics"].items()},
            # 学習内 / 学習外に分けた指標（extended/ の表だけ。合わせた指標の後ろに置く）
            **{k: fmt(v) for k, v in r["split"].items()},
            # 列名の数字は評価のシード。評価していないシードは空欄
            **{f"ep{s:02d}": fmt(r["stages"].get(s)) for s in seeds},
        }
        for r in rows
    ]
    long = [
        {
            **condition_cells(r),
            "episode_seed": s,
            "placement": "in" if s in IN_DIST_SEEDS else "ood",
            "stage": stage,
            "outcome": OUTCOMES[stage],
            "success": fmt(stage == 4),
            # 成功ならその時点、失敗なら最大ステップ数（400）
            "episode_steps": r["steps"][s],
        }
        for r in rows
        for s, stage in sorted(r["stages"].items())
    ]
    for name, table in (("runs.csv", wide), ("episodes_long.csv", long)):
        # utf-8-sig: 先頭に BOM を付ける。付けないと Excel が Shift_JIS と見なして日本語が化ける
        with (out_dir / name).open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(table[0].keys()))
            writer.writeheader()
            writer.writerows(table)

    print(f"評価 {len(rows)} 件 / エピソード {len(long)} 本")
    print(f"学習の条件の列: {', '.join(train_header(k) for k in train_cols)}")
    print(f"評価の条件の列: {', '.join(eval_cols)}")
    common = {**train_common, **eval_common}
    print("共通の設定（全条件で同じなので列にしていない）:")
    for key in sorted(common):
        print(f"    {key} = {common[key]}")
    print(f"saved -> {out_dir}/ (runs.csv, episodes_long.csv)")


def main() -> None:
    """runs/ の評価結果を全部集めて CSV に書く。"""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"), help="学習・評価の出力先")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/summary"), help="出力先")
    parser.add_argument(
        "--include-test", action="store_true", help="動作確認用の *_test の学習も含める"
    )
    args = parser.parse_args()

    rows, extended = collect(args.runs_dir, args.include_test)
    if not rows and not extended:
        raise SystemExit(f"{args.runs_dir}/*/eval_*/results.json が1つもない")
    if rows:
        write_csvs(rows, args.out_dir)
    if extended:
        print("\n評価の本数を増やした条件（既定の 50 本と合わせた行）:")
        write_csvs(extended, args.out_dir / "extended")


if __name__ == "__main__":
    main()
