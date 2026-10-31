#!/usr/bin/env bash
# 学習: ポリシーを学習する（処理の中身は src/pai_act/exps/train.py）。
#
# 使い方:
#   bash scripts/train.sh                            # 下に書いた設定で学習
#   bash scripts/train.sh --steps 5000 --name test   # 引数を足すと上書きできる（後に書いた値が優先される）
#   uv run python -m pai_act.exps.train --help       # 設定項目の一覧と既定値
#
# 出力: runs/<name>/（config.json / wandb のローカルファイル / チェックポイント ckpt_*.pt）
#       学習再開用: runs/<name>/lightning/last.ckpt（bash scripts/train.sh --resume <このパス>）
# 学習曲線: wandb の Web 画面（https://wandb.ai）。送信しないなら --wandb-mode offline / disabled

set -euo pipefail          # エラー・未定義変数・パイプ途中の失敗で止める
cd "$(dirname "$0")/.."    # どこから実行してもリポジトリ直下で動かす（data/ や runs/ は相対パスのため）

uv run python -m pai_act.exps.train \
    --name baseline \
    --policy baseline \
    "$@"
