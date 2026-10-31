"""実験の実行に関わるモジュール。

「(実行)」が付いたモジュールは `python -m pai_act.exps.<名前>` で直接実行できる。
普段は scripts/<名前>.sh から呼ぶ（リポジトリ直下に移動してから実行してくれる）。

部品:
    sim.py          gym-aloha の薄いラッパ（借用部分の物理シミュレータを扱うのはここだけ）
    config.py       ハイパーパラメータと実行設定（dataclass → コマンドライン引数）
    checkpoint.py   設定からのポリシーの組み立て、チェックポイントの保存・読み込み
    inference.py    推論時の行動チャンクの実行（action chunking / temporal ensembling）

実行:
    train.py        (実行) 学習ループ（全ポリシーで共通）
    evaluate.py     (実行) シミュレータでのロールアウト、成功率の算出、動画保存（全ポリシーで共通）
    replay_demo.py  (実行) デモの行動をシミュレータで再生し、データとシミュレータの対応を検証する
    check_env.py    (実行) 環境の動作確認（GPU / 描画 / 動画保存）
"""
