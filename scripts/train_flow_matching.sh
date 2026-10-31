#!/usr/bin/env bash
# 学習（Flow Matching）: Flow Matching ポリシーを学習する（処理の中身は src/pai_act/exps/train.py、モデルは policies/flow_matching.py）。
#
# 使い方:
#   bash scripts/train_flow_matching.sh                                   # 下に書いた設定で学習
#   bash scripts/train_flow_matching.sh --steps 300 --name flow_matching_test --wandb-mode disabled   # 動作確認
#   uv run python -m pai_act.exps.train --help                            # 設定項目の一覧と既定値
#
# 評価（ソルバーと推論ステップ数は学習し直さずに切り替えられる。出力は eval_ckpt_last_euler10/ などに分かれる）:
#   bash scripts/evaluate.sh --ckpt runs/flow_matching/ckpt_last.pt                              # 学習時の設定（Euler 10 ステップ）
#   bash scripts/evaluate.sh --ckpt runs/flow_matching/ckpt_last.pt --solver heun --num-inference-steps 5
#
# 比較の条件を揃えるため、学習率・バッチ・chunk・ステップ数・モデルの大きさは ACT（train_act.sh）・
# Diffusion（train_diffusion.sh）と同じにしている。

set -euo pipefail          # エラー・未定義変数・パイプ途中の失敗で止める
cd "$(dirname "$0")/.."    # どこから実行してもリポジトリ直下で動かす（data/ や runs/ は相対パスのため）

uv run python -m pai_act.exps.train \
    --name flow_matching \
    --policy flow_matching \
    --lr 1e-5 \
    --lr-backbone 1e-5 \
    --steps 80000 \
    --save-every 10000 \
    "$@"
