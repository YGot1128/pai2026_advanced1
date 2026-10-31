#!/usr/bin/env bash
# 実験結果の一覧表: runs/*/eval_*/results.json を全部集めて、1 つの CSV にまとめる
# （処理の中身は src/pai_act/visualize/summary.py）。新しい条件は評価すれば自動で行が増える。
#
# 使い方:
#   bash scripts/summarize.sh                  # 動作確認用の *_test を除いて集める
#   bash scripts/summarize.sh --include-test   # *_test も含める
#
# 出力: outputs/summary/
#   runs.csv           … 1 行 = 1 回の評価（条件・成功率・推論時間・ep00〜ep49 の段階 0〜4）
#   episodes_long.csv  … 1 行 = 1 エピソード（ピボットテーブル用）
#   extended/          … 評価の本数を増やした条件だけの同じ形の表（既定の 50 本と追加の評価を合わせて 1 行。
#                        例: シード 0〜49 と 50〜199 で 200 本）。上の 2 つには入れない。
#                        合わせた指標に加え、学習内（シード 0〜49、in_*）と学習外（50 以降、ood_*）に分けた列もある
# 既存の表を残したいときは --out-dir で出力先を変える（例: --out-dir outputs/summary_1025）
# 段階は最大報酬 0〜4（0: no_touch … 3: transfer_failed、4: 成功）。大きいほど成功に近い

set -euo pipefail          # エラー・未定義変数・パイプ途中の失敗で止める
cd "$(dirname "$0")/.."    # どこから実行してもリポジトリ直下で動かす（runs/ や outputs/ は相対パスのため）

uv run python -m pai_act.visualize.summary "$@"
