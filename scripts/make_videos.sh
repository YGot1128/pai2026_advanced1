#!/usr/bin/env bash
# 提出動画の素材: 評価で保存したロールアウトの動画を横に並べ、ラベルを付ける
# （処理の中身は src/pai_act/visualize/videos.py。素材の一覧は同じファイルの CLIPS）。
#
# 使い方:
#   bash scripts/make_videos.sh                          # すべての素材を作る
#   bash scripts/make_videos.sh --only no2_heads_seed3   # 1 本だけ作る
#   bash scripts/make_videos.sh --list                   # 素材の一覧と、元の動画が揃っているか
#
# 出力: outputs/video/（<名前>.mp4 と、使った動画・結果を書いた <名前>.sources.md）

set -euo pipefail          # エラー・未定義変数・パイプ途中の失敗で止める
cd "$(dirname "$0")/.."    # どこから実行してもリポジトリ直下で動かす（runs/ や outputs/ は相対パスのため）

uv run python -m pai_act.visualize.videos "$@"
