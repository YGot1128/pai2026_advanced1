#!/usr/bin/env bash
# 比較表とグラフ: 条件ごとの評価結果（results.json）を集めて、表（Markdown・CSV）と図（PNG）を作る
# （処理の中身は src/pai_act/visualize/compare.py）。
#
# 使い方:
#   bash scripts/compare.sh --list                     # プリセット（決めておいた条件の組）の一覧と、結果が揃っているか
#   bash scripts/compare.sh --preset act_ablation      # ACT 3 条件（1 層 / 7 層 / クエリ固定）
#   bash scripts/compare.sh --preset heads             # baseline / ACT / Diffusion
#   bash scripts/compare.sh --preset diffusion_steps   # Diffusion の推論ステップ数（D9）
#   bash scripts/compare.sh --preset diffusion_ensemble   # Diffusion の temporal ensembling の有無（D10）
#   bash scripts/compare.sh --run "ACT=runs/act/eval_ckpt_last" --run "X=runs/x/eval_ckpt_last" --out-dir outputs/compare/x
#
# 出力: outputs/compare/<プリセット名>/（table.md・table.csv・success_rate.png・outcomes.png・speed_vs_success.png）

set -euo pipefail          # エラー・未定義変数・パイプ途中の失敗で止める
cd "$(dirname "$0")/.."    # どこから実行してもリポジトリ直下で動かす（runs/ や outputs/ は相対パスのため）

uv run python -m pai_act.visualize.compare "$@"
