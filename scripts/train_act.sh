#!/usr/bin/env bash
# 学習（ACT）: ACT ポリシーを学習する（処理の中身は src/pai_act/exps/train.py、モデルは policies/act.py）。
#
# 使い方:
#   bash scripts/train_act.sh                                         # 下に書いた設定で学習
#   bash scripts/train_act.sh --steps 100 --name act_test --wandb-mode disabled   # 動作確認
#   uv run python -m pai_act.exps.train --help                        # 設定項目の一覧と既定値
#
# 評価: bash scripts/evaluate.sh --ckpt runs/act/ckpt_last.pt
#
# 設定は ACT 原著の sim 実験・LeRobot の ACT 設定に合わせている（モデルの大きさは exps/config.py の既定値）。
#   学習率 1e-5（バックボーンも 1e-5）/ バッチ 8 / chunk 100 / KL の重み 10
#   ステップ数は実施計画（PLAN_final_project.md）の 80k。LeRobot の既定は 100k

set -euo pipefail          # エラー・未定義変数・パイプ途中の失敗で止める
cd "$(dirname "$0")/.."    # どこから実行してもリポジトリ直下で動かす（data/ や runs/ は相対パスのため）

uv run python -m pai_act.exps.train \
    --name act \
    --policy act \
    --lr 1e-5 \
    --lr-backbone 1e-5 \
    --steps 80000 \
    --save-every 10000 \
    "$@"
