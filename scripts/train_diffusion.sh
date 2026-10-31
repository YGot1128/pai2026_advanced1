#!/usr/bin/env bash
# 学習（Diffusion）: Diffusion ポリシーを学習する（処理の中身は src/pai_act/exps/train.py、モデルは policies/diffusion.py）。
#
# 使い方:
#   bash scripts/train_diffusion.sh                                   # 下に書いた設定で学習
#   bash scripts/train_diffusion.sh --steps 300 --name diffusion_test --wandb-mode disabled   # 動作確認
#   uv run python -m pai_act.exps.train --help                        # 設定項目の一覧と既定値
#
# 評価（サンプラーと推論ステップ数は学習し直さずに切り替えられる。出力は eval_ckpt_last_ddim10/ などに分かれる）:
#   bash scripts/evaluate.sh --ckpt runs/diffusion/ckpt_last.pt                                  # 学習時の設定（DDIM 10 ステップ）
#   bash scripts/evaluate.sh --ckpt runs/diffusion/ckpt_last.pt --sampler ddim --num-inference-steps 20
#   bash scripts/evaluate.sh --ckpt runs/diffusion/ckpt_last.pt --sampler ddpm
#
# 比較の条件を揃えるため、学習率・バッチ・chunk・ステップ数・モデルの大きさは ACT（train_act.sh）と同じにしている。
# 拡散の設定（T = 100、cosine スケジュール、DDIM 10 ステップ）は Diffusion Policy（Chi et al. 2023）に合わせている。
# Diffusion Policy の学習率は 1e-4。損失が下がりにくければ候補にする（変えたら ACT との違いとして記録する）。

set -euo pipefail          # エラー・未定義変数・パイプ途中の失敗で止める
cd "$(dirname "$0")/.."    # どこから実行してもリポジトリ直下で動かす（data/ や runs/ は相対パスのため）

uv run python -m pai_act.exps.train \
    --name diffusion \
    --policy diffusion \
    --lr 1e-5 \
    --lr-backbone 1e-5 \
    --steps 80000 \
    --save-every 10000 \
    "$@"
