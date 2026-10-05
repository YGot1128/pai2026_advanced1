# ACT スクラッチ実装 × ALOHA シミュレーション

Physical AI 応用1講座 最終課題のリポジトリです。

模倣学習の手法 **ACT (Action Chunking Transformer)** を素の PyTorch でスクラッチ実装し、ALOHA 双腕ロボットのシミュレータ上でキューブの受け渡しタスク（transfer cube）を実行します。

> **進捗状況**: 開発中です。現在はデータの読み出しと、シミュレータとの対応の検証まで完了しています（[進捗](#進捗)を参照）。

---

## 概要

| 項目 | 内容 |
|---|---|
| タスク | **transfer cube**: 右腕でキューブを掴み、左腕に受け渡す |
| 手法 | ACT（CVAE + Transformer encoder-decoder + action chunking） |
| データ | [`lerobot/aloha_sim_transfer_cube_human`](https://huggingface.co/datasets/lerobot/aloha_sim_transfer_cube_human)（人間による遠隔操作のデモ 50本 / 50fps / 1本400フレーム） |
| シミュレータ | [gym-aloha](https://github.com/huggingface/gym-aloha)（MuJoCo / dm_control） |
| 観測 | 上方カメラ画像 480×640 RGB + 両腕の関節角 14次元 |
| 行動 | 両腕の目標関節角 14次元（各腕: 6関節 + グリッパー） |
| 評価指標 | シミュレータ上でのタスク成功率（reward = 4 に到達したエピソードの割合） |

### この課題のポイント

- **ポリシーをスクラッチで実装**: Multi-Head Attention から CVAE の損失関数、推論時の temporal ensembling までを自前で書いています。既存フレームワークのポリシー実装は使いません。
- **アクションヘッドの比較（予定）**: 同じバックボーン・同じデータ・同じ評価条件で、`CVAE（ACT 原著）` / `Diffusion (DDPM)` / `Flow Matching` の3種類を比較します。3種類とも自前で実装するため、条件を揃えた公平な比較ができます。
- **ROS2 ノード化（予定）**: 学習済みポリシーを ROS2 Jazzy のノードとして動かします。

## スクラッチ実装と借用部分の線引き

原則として、**学習・推論の中身に関わる部分はすべて自前で実装**します。借用するのは物理シミュレータ、ファイル I/O、事前学習済みの画像バックボーンだけです。

| 自前で実装する | 借用する |
|---|---|
| Transformer（Multi-Head Attention を含む。`nn.Transformer` / `nn.MultiheadAttention` は使わない） | 物理シミュレータ（gym-aloha / MuJoCo） |
| 位置エンコーディング（画像用の2D版・系列用の1D版） | データセットのダウンロードと読み出し（parquet / 動画デコード） |
| ACT 本体（CVAE エンコーダ + encoder-decoder + 行動チャンクの出力ヘッド） | ResNet18 の ImageNet 事前学習重み |
| CVAE の学習目的関数（再構成損失 + KL項、reparameterization trick） | PyTorch の基本演算・autograd |
| action chunking / temporal ensembling | |
| 観測・行動の正規化、学習ループ、評価ハーネス | |
| 比較用の Diffusion / Flow Matching ヘッド | |

## 進捗

- [x] 開発環境の構築（devcontainer + uv、WSL2 上で MuJoCo を GPU 描画）
- [x] データセットの読み出し
- [x] デモの行動をシミュレータで再生し、データとシミュレータの対応を検証
- [ ] 学習 → ロールアウト → 動画保存のパイプライン
- [ ] ACT の実装と学習
- [ ] Diffusion / Flow Matching ヘッドとの比較実験
- [ ] ROS2 ノード化

## 検証結果: デモの再生

モデルを学習する前に、**データセットの行動をシミュレータでそのまま再生すれば成功するか**を確かめました。ここで成功しなければ、学習したモデルがいくら正しくても成功率は上がらないためです。

### 課題: キューブの初期位置がデータに記録されていない

gym-aloha はリセットのたびにキューブの位置を乱数で決めます。一方、データセットにはキューブの位置が保存されていません。そこで次の手順で位置を推定しました。

1. シミュレータでキューブを 7×7 の格子点に置いて描画し、赤いキューブの画素重心を求める
2. 画素重心からテーブル上の (x, y) への写像を2次多項式で当てはめる（当てはめ誤差は最大 0.92 mm）
3. 各デモの初期フレームでキューブの画素重心を求め、2 の写像で位置を推定してシミュレータに配置する

### 結果

| 指標 | 結果 |
|---|---|
| 成功率 | **50 / 50（100%）** |
| 腕関節の追従誤差（データの状態との平均絶対誤差、エピソードごと） | 0.0001〜0.0031 rad |
| 推定したキューブ位置の画素ずれ | 0.5 px 以下 |
| 成功に到達したステップ | 最小 217 / 中央値 290 / 最大 396 |

これで、データの行動空間とシミュレータの行動空間が一致していること、および成功判定が正しく機能していることを確認できました。

### 評価条件についての注意

gym-aloha の既定のエピソード長は **300ステップ（6秒）** です。しかし人間のデモでは **50本中19本が300ステップ以降に成功** しています。既定の長さのままでは、正しいデモでも約4割が「失敗」と判定されてしまいます。

このリポジトリでは、評価のエピソード長をデモと同じ **400ステップ（8秒）** にしています。

## リポジトリ構成

`（予定）` が付いたファイルはまだ作成していません。

```
.
├── .devcontainer/             # 開発環境（Ubuntu 24.04 + uv + MuJoCo の EGL 描画）
├── src/pai_act/
│   ├── data/
│   │   ├── dataset.py         # LeRobot v3.0 形式データセットの読み出し【借用: ファイルI/O】
│   │   ├── chunk_dataset.py   # 学習サンプルの切り出し（観測1フレーム + 行動kステップ）（予定）
│   │   └── normalize.py       # 統計量の算出と正規化（予定）
│   ├── nn/                    # タスクに依存しない部品
│   │   ├── attention.py       # Multi-Head Attention（予定）
│   │   ├── transformer.py     # Encoder / Decoder 層（予定）
│   │   ├── position.py        # sinusoidal 位置エンコーディング（1D / 2D）（予定）
│   │   └── backbone.py        # ResNet18 と特徴の射影【借用: 事前学習重み】（予定）
│   ├── policies/              # 比較実験の対象
│   │   ├── base.py            # 共通インタフェース（compute_loss / predict_chunk）（予定）
│   │   ├── trunk.py           # 全手法で共有する部分: 画像・状態 → 観測トークン列（予定）
│   │   ├── act.py             # CVAE ヘッド（ACT 原著）（予定）
│   │   ├── diffusion.py       # DDPM ヘッド（予定）
│   │   └── flow_matching.py   # Flow Matching ヘッド（予定）
│   └── exps/                  # 実験の実行に関わるもの
│       ├── sim.py             # gym-aloha の薄いラッパ【借用: 物理シム】
│       ├── inference.py       # action chunking / temporal ensembling（予定）
│       ├── train.py           # 学習ループ（全手法で共通）（予定）
│       ├── evaluate.py        # ロールアウト・成功率・動画保存（全手法で共通）（予定）
│       └── config.py          # ハイパーパラメータ（予定）
├── scripts/                   # 実行の入口（処理の中身は src/pai_act に置く）
│   ├── check_env.py           # 環境の動作確認（GPU / 描画 / 動画保存）
│   └── replay_demo.py         # デモをシミュレータで再生して検証する
├── tests/                     # 自作部品の単体テスト（予定）
├── ros2/pai_act_ros/          # ROS2 パッケージ（colcon でビルド）（予定）
├── pyproject.toml / uv.lock
└── PLAN_final_project.md      # 実施計画
```

### 構成の考え方

- **公平な比較を構造で保証する**: 画像・状態を観測トークン列にする部分（`policies/trunk.py`）を3手法で共有し、違いはヘッドだけにしています。全ポリシーが `policies/base.py` の共通インタフェースを実装するので、学習ループ（`exps/train.py`）と評価（`exps/evaluate.py`）は1つずつです。
- **借用部分をファイル単位で閉じ込める**: 借用は `data/dataset.py`、`nn/backbone.py`、`exps/sim.py` の3ファイルに限っています。上の「スクラッチ実装と借用部分の線引き」の表と対応しています。
- **参考にした構成**: 部品の分け方（backbone / transformer / position encoding）は [DETR](https://github.com/facebookresearch/detr) と、それを流用した [ACT 公式実装](https://github.com/tonyzhaozh/act) を参考にしました。手法ごとにポリシーを分けて共通インタフェースを持たせる形は、[LeRobot](https://github.com/huggingface/lerobot) と [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) を参考にしています。参考にしたのは構成だけで、コードは自前で書いています。

## 環境構築

### 動作環境

- Windows + WSL2（Ubuntu 24.04）+ Docker + VS Code Dev Containers
- NVIDIA GPU（開発には RTX 4090 Laptop 16GB を使用）
- Python 3.12 / PyTorch 2.14（CUDA 13.0）/ gym-aloha 0.1.4

### セットアップ

1. VS Code でこのリポジトリを開き、「Reopen in Container」を実行します。コンテナの作成時に `uv sync` で依存ライブラリが入ります。
2. 環境の動作確認をします。

   ```bash
   uv run python scripts/check_env.py
   ```

   `renderer:` に NVIDIA の GPU が表示されれば、GPU で描画できています。`llvmpipe` と表示される場合は CPU 描画になっており、約6倍遅くなります。

> **WSL2 で MuJoCo を GPU 描画するための設定**: `/dev/dxg` と `/usr/lib/wsl` をコンテナに渡し、Mesa の D3D12 ドライバ経由で EGL 描画を GPU に載せています。設定は [.devcontainer/devcontainer.json](.devcontainer/devcontainer.json) を参照してください。

## 実行方法

```bash
# デモの再生検証（全50本）。初回のみデータセットのダウンロードと動画のデコード（約18GB のキャッシュ作成）が走る
uv run python scripts/replay_demo.py

# 一部のエピソードだけ再生し、比較動画（左: シム / 右: デモ）を保存する
uv run python scripts/replay_demo.py --episodes 0 1 2 --video-episodes 0
```

出力は `outputs/` に保存されます。

## 出典・参考文献

- T. Z. Zhao, V. Kumar, S. Levine, C. Finn. *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware.* RSS 2023. [arXiv:2304.13705](https://arxiv.org/abs/2304.13705)（ACT の原著論文）
- データセット: [lerobot/aloha_sim_transfer_cube_human](https://huggingface.co/datasets/lerobot/aloha_sim_transfer_cube_human)（MIT License）
- シミュレータ: [huggingface/gym-aloha](https://github.com/huggingface/gym-aloha)（Apache License 2.0）
- 到達目標の参考値: [lerobot/act_aloha_sim_transfer_cube_human](https://huggingface.co/lerobot/act_aloha_sim_transfer_cube_human)

## ライセンス

未定
