#!/usr/bin/env bash
# 環境の動作確認: GPU(PyTorch)・MuJoCo の GPU 描画・gym-aloha の動画保存が通るかを見る
# （処理の中身は src/pai_act/exps/check_env.py）。
#
# 使い方: bash scripts/check_env.sh
# 出力: outputs/check_env.mp4。最後に "ALL OK" と出れば成功。

set -euo pipefail          # エラー・未定義変数・パイプ途中の失敗で止める
cd "$(dirname "$0")/.."    # どこから実行してもリポジトリ直下で動かす（outputs/ は相対パスのため）

uv run python -m pai_act.exps.check_env
