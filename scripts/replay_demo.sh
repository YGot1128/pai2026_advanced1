#!/usr/bin/env bash
# デモの再生検証: データセットの行動をシミュレータで再生し、データとシミュレータの対応を確かめる
# （処理の中身は src/pai_act/exps/replay_demo.py）。
# 初回のみデータセットのダウンロードと動画のデコード（約18GB のキャッシュ作成）が走る。
#
# 使い方:
#   bash scripts/replay_demo.sh                                      # 全50本を再生
#   bash scripts/replay_demo.sh --episodes 0 1 2 --video-episodes 0  # 一部だけ再生し、比較動画も保存
#
# 出力: outputs/replay/replay_epXX.mp4（左: シミュレータでの再生 / 右: デモ動画）

set -euo pipefail          # エラー・未定義変数・パイプ途中の失敗で止める
cd "$(dirname "$0")/.."    # どこから実行してもリポジトリ直下で動かす（data/ や outputs/ は相対パスのため）

uv run python -m pai_act.exps.replay_demo "$@"
