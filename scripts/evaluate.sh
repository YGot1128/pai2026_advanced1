#!/usr/bin/env bash
# 評価: 学習済みポリシーをシミュレータでロールアウトし、成功率を測って動画を保存する
# （処理の中身は src/pai_act/exps/evaluate.py）。
#
# 使い方:
#   bash scripts/evaluate.sh                                   # 下に書いたチェックポイントを評価
#   bash scripts/evaluate.sh --ckpt runs/xxx/ckpt_last.pt      # 別のチェックポイントを評価
#   bash scripts/evaluate.sh --num-episodes 10 --no-temporal-ensemble
#   uv run python -m pai_act.exps.evaluate --help              # 設定項目の一覧と既定値
#
# 出力: runs/<name>/eval_<ckpt名>/（results.json と videos/*.mp4。動画は先頭5本と失敗した全エピソード）
# results.json の項目と失敗の型の定義: docs/EVALUATION.md

set -euo pipefail          # エラー・未定義変数・パイプ途中の失敗で止める
cd "$(dirname "$0")/.."    # どこから実行してもリポジトリ直下で動かす（data/ や runs/ は相対パスのため）

uv run python -m pai_act.exps.evaluate \
    --ckpt runs/baseline/ckpt_last.pt \
    --num-episodes 50 \
    "$@"
