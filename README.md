# ACT スクラッチ実装 × ALOHA シミュレーション

Physical AI 応用1講座 最終課題のリポジトリです。

模倣学習の手法 **ACT (Action Chunking Transformer)** を素の PyTorch でスクラッチ実装し、ALOHA 双腕ロボットのシミュレータ上でキューブの受け渡しタスク（transfer cube）を実行します。

> **進捗状況**: 開発中です。ACT をスクラッチ実装して学習・評価し、成功率 **72%（36/50）** を得ました（[結果: ACT](#結果-act)）。現在は追加実験と、比較用の Diffusion / Flow Matching ヘッドの実装を進めています（[進捗](#進捗)を参照）。

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

原則として、**学習・推論の中身に関わる部分はすべて自前で実装**します。借用するのは物理シミュレータ、ファイル I/O、事前学習済みの画像バックボーンだけです。学習ループの定型処理には PyTorch Lightning を使いますが、ポリシー（モデル）は Lightning に依存しない素の PyTorch の `nn.Module` で、Lightning は `exps/train.py` の中だけで使います。

| 自前で実装する | 借用する |
|---|---|
| Transformer（Multi-Head Attention を含む。`nn.Transformer` / `nn.MultiheadAttention` は使わない） | 物理シミュレータ（gym-aloha / MuJoCo） |
| 位置エンコーディング（画像用の2D版・系列用の1D版） | データセットのダウンロードと読み出し（parquet / 動画デコード） |
| ACT 本体（CVAE エンコーダ + encoder-decoder + 行動チャンクの出力ヘッド） | ResNet18 の ImageNet 事前学習重み |
| CVAE の学習目的関数（再構成損失 + KL項、reparameterization trick） | PyTorch の基本演算・autograd |
| action chunking / temporal ensembling | 学習ループの定型処理（PyTorch Lightning: 逆伝播・更新の呼び出し、混合精度、ログ、保存） |
| 観測・行動の正規化、学習ステップの中身（損失・optimizer の設定）、評価ハーネス | |
| 比較用の Diffusion / Flow Matching ヘッド | |

## 生成 AI（Claude Code）の使い方と分担

この課題は ACT を学ぶことも目的にしているため、開発には Claude Code を使いつつ、**手法の中核は自分で実装する**と決めて分担しました。
上の表の「自前で実装する」は「ライブラリから借りずに書く」という意味で、その中の誰が書いたかの分担は次の表のとおりです。
分担の約束は [CLAUDE.md](CLAUDE.md)（Claude Code への指示書）に書き、全セッションで守らせています。
Claude が関わったコミットには `Co-Authored-By: Claude` が付いています。

| 担当 | 内容 |
|---|---|
| 自分 | 手法の中核の実装: Multi-Head Attention（`nn/attention.py`）、位置エンコーディング（`nn/position.py`）、Transformer の Encoder / Decoder 層（`nn/transformer.py`）、観測トークン列の作成（`policies/trunk.py`）、ACT 本体と CVAE の損失（`policies/act.py`）、Diffusion の共有デノイザ（`policies/denoiser.py`）、ノイズスケジュール・損失・DDPM / DDIM サンプラー（`policies/diffusion.py`） |
| Claude Code | 上のファイルの骨組み（docstring・テンソル形状のコメント・手順の Step コメント）と単体テスト（`tests/`）、学習・評価・推論の配線（`exps/`・`scripts/`）、データの読み出しと正規化（`data/`）、開発環境（`.devcontainer/`）、ドキュメントの下書き |

進め方:

1. 実装の前に、各ステップの考え方・数式・数値例を Claude から解説してもらう
2. 自分で実装し、単体テストを通す。Claude はこれらのファイルの中身を書かない（エラーのときは原因の説明とヒントだけを受ける）
3. 理解チェックの問いに自分の言葉で答え、Claude の添削を受けて自分で直す。記録は [ACT_IMPLEMENTATION.md](ACT_IMPLEMENTATION.md) と [DIFFUSION_IMPLEMENTATION.md](DIFFUSION_IMPLEMENTATION.md) に残している
4. 書いたコードのレビューを受ける（バグ・形状の誤り・原著や公式実装との違い）

## 進捗

- [x] 開発環境の構築（devcontainer + uv、WSL2 上で MuJoCo を GPU 描画）
- [x] データセットの読み出し
- [x] デモの行動をシミュレータで再生し、データとシミュレータの対応を検証
- [x] 学習 → ロールアウト → 動画保存のパイプライン（仮モデルで一周を確認）
- [x] ACT の実装と学習（成功率 72%）
- [ ] ACT の成功率を 80% 台へ（学習シードを変えたばらつきの確認を含む）
- [ ] ACT の Decoder 層数（1層 / 7層）の比較（公式実装のバグで実質1層になっている点の検証）
- [ ] ACT の行動クエリの位置ベクトル（学習 / 固定 sinusoidal）の比較（論文の記述と公式実装の食い違いの検証）
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

## パイプライン: 学習 → ロールアウト → 動画保存

ACT 本体を実装する前に、**仮モデル**（ResNet18 + MLP で行動チャンクを直接回帰する [baseline](src/pai_act/policies/baseline.py)）を使って、学習から評価までを一通り繋げました。ポリシーは共通インタフェース（[policies/base.py](src/pai_act/policies/base.py)）を実装していれば何でも差し替えられます。ACT などの手法は、ポリシーのファイルを追加するだけで同じ学習・評価コードに載ります。

```
【学習】 scripts/train.sh → exps/train.py
  AlohaDataset（50本のデモ）
    └─ エピソード単位で 学習45本 / 検証5本 に分割
  ChunkDataset: 時刻 t の (画像, 関節角) と、t から k=100 ステップ分の行動チャンクを1サンプルにする
    └─ DataLoader（バッチ8）→ policy.compute_loss → AdamW で更新（PyTorch Lightning の Trainer で回す）
  出力: runs/<name>/ckpt_*.pt（設定・重み・正規化の統計量をまとめて保存）
        runs/<name>/lightning/last.ckpt（学習再開用。optimizer の状態も含む）

【評価】 scripts/evaluate.sh → exps/evaluate.py
  チェックポイントからポリシーを復元
  シード 0..49 でシミュレータをリセット（キューブの位置がシードで決まる）し、閉ループで実行:
    観測 → policy.predict_chunk → ChunkExecutor（action chunking / temporal ensembling）→ env.step → 観測 …
  出力: results.json（成功率・到達段階・失敗の型）と videos/*.mp4（左上にシード・ステップ・報酬を表示）
```

results.json の各項目と、報酬・失敗の型の定義は [docs/EVALUATION.md](docs/EVALUATION.md) にまとめています。

学習と推論で前処理がずれないよう、正規化はポリシーの内部で行い、統計量もポリシーの重みと一緒に保存しています。ポリシーが外とやり取りする値は、どれもシミュレータと同じ単位です。

### 仮モデルでの結果

既定の設定（chunk k=100、バッチ8、20,000ステップ、temporal ensembling あり）で学習・評価しました。学習は RTX 4090 Laptop で約10分、50エピソードの評価は約5分です。

| 指標 | 結果 |
|---|---|
| 成功率（50エピソード） | **30 / 50（60%）** |
| 各段階まで到達した割合 | キューブに触れた 92% / 持ち上げた 80% / 受け渡しを試みた 60% / 成功 60% |
| 失敗20本の内訳（最大報酬） | 触れない 4本 / 触れたが持ち上げられない 6本 / 持ち上げたが受け渡せない 10本 |
| 損失（L1、正規化後） | 学習 0.67 → 0.13 / 検証 0.47（1kステップ）→ 0.25（20kステップ）。検証損失は 8k ステップ以降ほぼ横ばい |

これで、データの読み出しから成功率・動画の出力までが繋がっていることを確認できました。この数値は学習1回・評価シード 0〜49 での結果で、ばらつきはまだ測っていません。ACT との比較で同じ評価条件を使えるよう、仮モデルもベースラインとして残しておきます。

失敗のうち最も多いのは、持ち上げた後の受け渡しです。仮モデルは画像特徴を空間方向に平均しているため、キューブとグリッパーの位置関係のような細かい空間情報が失われることが一因と考えています。ACT では、画像特徴を位置情報付きのトークン列として Transformer で扱います。

## 結果: ACT

スクラッチ実装した ACT（[policies/act.py](src/pai_act/policies/act.py)）を学習し、仮モデルと同じ条件で評価しました。評価条件はシード 0〜49 の 50 エピソード、エピソード長 400 ステップ、temporal ensembling ありです。

### 学習の設定

主な値は ACT 原著・公式実装に合わせました。設定は [scripts/train_act.sh](scripts/train_act.sh) にあります。

| 項目 | 値 |
|---|---|
| 行動チャンク k | 100 |
| Transformer | d_model 512 / head 8 / FF 3200 / dropout 0.1 |
| 層数 | Encoder 4 / Decoder 1 / CVAE エンコーダ 4 |
| CVAE | 潜在変数 32 次元、KL の重み β = 10、推論時は z = 0 |
| 行動クエリの位置ベクトル | 学習（公式実装と同じ） |
| 最適化 | AdamW、学習率 1e-5（バックボーンも 1e-5）、weight decay 1e-4、勾配クリップ 10 |
| 学習 | バッチ 8、80,000 ステップ、混合精度（AMP） |
| パラメータ数 | 51.6M |
| 学習時間 | 1 時間 26 分（RTX 4090 Laptop、15.5 it/s） |

Decoder を 1 層にしたのは、原著の設定（7 層）に対して、公式実装ではバグのため 1 層目の出力しか使われていないからです（[tonyzhaozh/act#25](https://github.com/tonyzhaozh/act/issues/25)）。原著の報告値は実質 1 層で出たものと考え、LeRobot と同じく 1 層を既定値にしています。7 層すべてを使った場合との比較は、追加実験として行う予定です。

### 結果（仮モデルとの比較）

| 指標 | 仮モデル（baseline） | ACT |
|---|---|---|
| 成功率（50エピソード） | 30 / 50（60%） | **36 / 50（72%）** |
| 各段階まで到達した割合（触れた / 持ち上げた / 受け渡しを試みた） | 92% / 80% / 60% | 96% / 90% / 72% |
| 失敗の内訳（`no_touch` / `not_lifted` / `not_transferred` / `transfer_failed`） | 4 / 6 / 10 / 0 本 | 2 / 3 / 9 / 0 本 |
| 成功までのステップ（最小 / 中央値 / 最大） | 231 / 303 / 372 | 233 / 305 / 395 |
| 検証 L1（最終） | 0.25（20k ステップ） | 0.254（80k ステップ） |
| `predict_chunk` 1回の時間（平均） | — | 7.1 ms |

失敗の型の定義は [docs/EVALUATION.md](docs/EVALUATION.md) にあります。検証データは両者で同じ 5 エピソードです（学習シード 0 で分割）。

ACT の数値は、失敗した全エピソードの動画を保存するように評価コードを直した後の、2 回目の評価のものです。1 回目の評価（同じチェックポイント・同じシード）では、成功率は同じ 72% でしたが、到達段階は 96% / 84% / 72%、失敗の内訳は 2 / 6 / 6 / 0 本でした（下の「評価の再現性」を参照）。

### 学習曲線

| ステップ | 検証 L1 | 検証 KL |
|---|---|---|
| 1k | 0.362 | 0.013 |
| 30k | 0.258 | 0.006 |
| 80k | 0.254 | 0.009 |

- 検証 L1 は 30k ステップあたりから約 0.255 で横ばいになりました。学習 L1 は最後に 0.050 まで下がっていて検証との差は大きいものの、検証 L1 は悪化していません。
- 検証 KL（32 次元の和）は、学習を通して 0.002〜0.013 とほぼ 0 のままでした（学習 KL も最後は 0.006）。

### 失敗の観察

2 回目の評価では、失敗した 14 本すべての動画を `runs/act/eval_ckpt_last/videos/` に保存しました（ファイル名の末尾が失敗の型）。

| 失敗の型 | 本数 | seed |
|---|---|---|
| `no_touch` | 2 | 39, 49 |
| `not_lifted` | 3 | 25, 29, 35 |
| `not_transferred` | 9 | 2, 5, 13, 21, 22, 27, 42, 44, 46 |

1 回目の評価で動画を確認できた失敗は seed 2 の 1 本だけでした（このときは `not_lifted`）。指がキューブの表面に触れた後、指を開かないままアームを深さ方向に下げようとして、キューブを掴めていませんでした。失敗の型ごとの動きは、2 回目の動画で確認中です。

### 評価の再現性

同じチェックポイントを同じシード 0〜49 で 2 回評価したところ、結果が完全には一致しませんでした。

- 50 本中 24 本で、終了までのステップ数か最大報酬が変わりました。ほとんどは成功までのステップが 1〜6 ずれただけです。
- 成否が入れ替わったエピソードが 2 本ありました（seed 7: 失敗 → 成功、seed 13: 成功 → 失敗）。成功率はたまたま同じ 72% でした。
- 最大報酬が 1 → 2 に変わったエピソードが 3 本あり（seed 2, 5, 22）、失敗の内訳は大きく変わりました（`not_lifted` 6 → 3 本、`not_transferred` 6 → 9 本）。

シミュレータの初期配置はシードで決まるので、違いはポリシーの推論から来ていると考えています。GPU の計算は、cuDNN が選ぶアルゴリズム（`cudnn.benchmark`）などによって、浮動小数点の丸め誤差の出方が実行ごとに変わることがあります。閉ループでは、そのわずかな行動の差が次の観測を変え、接触の有無のような離散的な結果まで変えてしまいます。そのため、1 回の評価での失敗の内訳や、シードごとの成否には、学習のばらつきとは別に評価自体のばらつきが含まれます。

### まとめ

- **自前の ACT で成功率 72% が出ました。** すべての段階の到達率で仮モデルを上回っています。差が大きいのは持ち上げるまでの段階で、持ち上げた割合は 80% から 90% に上がりました。
- **ただし、仮モデルとの差（12 ポイント）は偶然の範囲に収まる大きさです。** Fisher の正確検定で p ≈ 0.29、95% 信頼区間（Wilson）は ACT が 58〜83%、仮モデルが 46〜72% で重なっています。どちらも学習 1 回・評価 50 回の結果です。さらに、同じチェックポイントを評価し直すだけでも、成否が変わるエピソードがありました（「評価の再現性」を参照）。学習シードを変えた追加の学習で、ばらつきを確かめる必要があります。
- **最も多い失敗は、持ち上げた後に受け渡しへ入れないことです。** 失敗 14 本のうち 9 本が `not_transferred` で、仮モデル（20 本中 10 本）と同じ傾向です。一方で、「受け渡しを試みた」と「成功」の割合はどちらも 72% で、左グリッパーがキューブに触れたエピソードはすべて成功しました。ACT は掴んで持ち上げるところまでは改善しましたが、持ち上げたキューブを受け渡し位置まで運ぶ段階が残った課題です。
- **CVAE の潜在変数 z は、ほとんど情報を持っていない可能性があります。** KL がほぼ 0 ということは、エンコーダの出力 N(μ, σ²) が事前分布 N(0, I) とほぼ一致していることを意味します。β = 10 では KL の項が強く効き、Decoder は z をほとんど使わずに観測だけから行動を出している（決定的な回帰に近い）と考えられます。z を事前分布からサンプルして出力がどれだけ変わるかを調べれば確かめられます。
- **検証 L1 は成功率の目安になりませんでした。** 検証 L1 は仮モデルと ACT でほぼ同じ（0.25）なのに、成功率は 12 ポイント違います。L1 は行動を 1 ステップずつ比べるだけで、閉ループで誤差が積み重なったときの振る舞いや、掴めたかどうかを反映しないためと考えています。
- **推論は制御周期に十分収まります。** `predict_chunk` 1 回は平均 6〜7 ms（1 回目 6.1 ms / 2 回目 7.1 ms）で、50 Hz 制御の 1 ステップ（20 ms）より短いです。
- 計画の目標（80% 台）にはまだ届いていません。次は、Decoder 層数（1 層 / 7 層）と行動クエリの位置ベクトル（学習 / 固定）の比較実験、および学習シードを変えた再学習を行います。

## リポジトリ構成

`（予定）` が付いたファイルはまだ作成していません。`（実装中）` は骨組みとテストだけがあり、中身を実装している途中のファイルです。

```
.
├── .devcontainer/             # 開発環境（Ubuntu 24.04 + uv + MuJoCo の EGL 描画）
├── src/pai_act/
│   ├── data/
│   │   ├── dataset.py         # LeRobot v3.0 形式データセットの読み出し【借用: ファイルI/O】
│   │   ├── chunk_dataset.py   # 学習サンプルの切り出し（観測1フレーム + 行動kステップ）
│   │   └── normalize.py       # 統計量の算出と正規化
│   ├── nn/                    # タスクに依存しない部品
│   │   ├── attention.py       # Multi-Head Attention
│   │   ├── transformer.py     # Encoder / Decoder 層
│   │   ├── position.py        # sinusoidal 位置エンコーディング（1D / 2D）
│   │   └── backbone.py        # ResNet18【借用: 事前学習重み】
│   ├── policies/              # 比較実験の対象
│   │   ├── base.py            # 共通インタフェース（compute_loss / predict_chunk）
│   │   ├── baseline.py        # パイプライン検証用の仮モデル（ResNet18 + MLP）
│   │   ├── trunk.py           # 全手法で共有する部分: 画像・状態 → 観測トークン列
│   │   ├── act.py             # CVAE ヘッド（ACT 原著）
│   │   ├── denoiser.py        # Diffusion / Flow Matching の共有デノイザ（自作 Transformer を再利用）（実装中）
│   │   ├── diffusion.py       # Diffusion ヘッド（DDPM で学習、DDPM / DDIM でサンプリング）（実装中）
│   │   └── flow_matching.py   # Flow Matching ヘッド（予定）
│   └── exps/                  # 実験の実行に関わるもの（★ は python -m で直接実行できる）
│       ├── sim.py             # gym-aloha の薄いラッパ【借用: 物理シミュレータ】
│       ├── inference.py       # action chunking / temporal ensembling
│       ├── checkpoint.py      # ポリシーの組み立て・チェックポイントの保存と読み込み
│       ├── config.py          # ハイパーパラメータ（dataclass からコマンドライン引数を自動生成）
│       ├── train.py           # ★ 学習ループ（全手法で共通、PyTorch Lightning）
│       ├── evaluate.py        # ★ ロールアウト・成功率・動画保存（全手法で共通）
│       ├── replay_demo.py     # ★ デモをシミュレータで再生して検証する
│       └── check_env.py       # ★ 環境の動作確認（GPU / 描画 / 動画保存）
├── scripts/                   # 実行用のシェルスクリプト（★ のモジュールを実験の設定付きで呼ぶ）
│   ├── train.sh               # baseline の学習
│   ├── train_act.sh           # ACT の学習
│   ├── train_diffusion.sh     # Diffusion の学習
│   ├── evaluate.sh
│   ├── replay_demo.sh
│   └── check_env.sh
├── docs/EVALUATION.md         # 評価の出力（results.json・報酬・失敗の型・動画）の読み方
├── tests/                     # 自作部品の単体テスト（データ切り出し・チャンク実行・保存と復元・ACT / Diffusion の各部品）
├── ros2/pai_act_ros/          # ROS2 パッケージ（colcon でビルド）（予定）
├── pyproject.toml / uv.lock
├── CLAUDE.md                  # Claude Code（AI）との作業分担の方針
├── ACT_IMPLEMENTATION.md      # ACT 実装の手順と進捗のチェックリスト
├── DIFFUSION_IMPLEMENTATION.md # Diffusion ヘッド実装の手順と進捗のチェックリスト
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
- Python 3.12 / PyTorch 2.14（CUDA 13.0）/ PyTorch Lightning 2.6 / gym-aloha 0.1.4

### セットアップ

1. VS Code でこのリポジトリを開き、「Reopen in Container」を実行します。コンテナの作成時に `uv sync` で依存ライブラリが入ります。
2. 環境の動作確認をします。

   ```bash
   bash scripts/check_env.sh
   ```

   `renderer:` に NVIDIA の GPU が表示されれば、GPU で描画できています。`llvmpipe` と表示される場合は CPU 描画になっており、約6倍遅くなります。

> **WSL2 で MuJoCo を GPU 描画するための設定**: `/dev/dxg` と `/usr/lib/wsl` をコンテナに渡し、Mesa の D3D12 ドライバ経由で EGL 描画を GPU に載せています。設定は [.devcontainer/devcontainer.json](.devcontainer/devcontainer.json) を参照してください。

## 実行方法

実行は `scripts/` のシェルスクリプトから行います。中では、リポジトリ直下に移動してから `uv run python -m pai_act.exps.<名前>` を呼んでいます。スクリプトの後ろに付けた引数はそのまま Python 側に渡り、スクリプト内の設定より優先されます。

```bash
# デモの再生検証（全50本）。初回のみデータセットのダウンロードと動画のデコード（約18GB のキャッシュ作成）が走る
bash scripts/replay_demo.sh

# 一部のエピソードだけ再生し、比較動画（左: シミュレータ / 右: デモ）を保存する
bash scripts/replay_demo.sh --episodes 0 1 2 --video-episodes 0
```

出力は `outputs/` に保存されます。

### 学習と評価

```bash
# 学習（出力: runs/<name>/ に config.json・wandb のローカルファイル・チェックポイント・学習再開用の lightning/last.ckpt）
# 学習曲線は wandb（https://wandb.ai）のプロジェクト pai-act に記録される（要 WANDB_API_KEY）
bash scripts/train.sh

# 設定を上書きして学習する
bash scripts/train.sh --name test --steps 5000

# 中断した学習を再開する（最後に save_every の倍数で保存した時点から。--name などの設定は中断前と揃える）
bash scripts/train.sh --resume runs/baseline/lightning/last.ckpt

# ACT を学習する（出力: runs/act/）。評価は bash scripts/evaluate.sh --ckpt runs/act/ckpt_last.pt
bash scripts/train_act.sh

# Diffusion を学習する（出力: runs/diffusion/）。評価ではサンプラーと推論ステップ数を学習し直さずに切り替えられる
# （出力先は runs/diffusion/eval_ckpt_last_ddim10/ のようにサンプラーごとに分かれる）
bash scripts/train_diffusion.sh
bash scripts/evaluate.sh --ckpt runs/diffusion/ckpt_last.pt --sampler ddim --num-inference-steps 10

# wandb に送らず試す（offline: 手元に保存のみ、後で wandb sync で送れる / disabled: 記録しない）
bash scripts/train.sh --name test --steps 100 --wandb-mode disabled

# 評価: 50エピソードをロールアウトして成功率を測り、先頭5エピソードと失敗した全エピソードの動画を保存する
# （出力: runs/<name>/eval_<ckpt名>/results.json と videos/*.mp4。項目と失敗の型の定義は docs/EVALUATION.md）
bash scripts/evaluate.sh

# 別のチェックポイントを、temporal ensembling を使わず（チャンクを k ステップずつそのまま実行して）評価する
bash scripts/evaluate.sh --ckpt runs/test/ckpt_last.pt --no-temporal-ensemble

# 設定項目の一覧と既定値
uv run python -m pai_act.exps.train --help
uv run python -m pai_act.exps.evaluate --help

# 単体テスト
uv run pytest
```

## 出典・参考文献

- T. Z. Zhao, V. Kumar, S. Levine, C. Finn. *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware.* RSS 2023. [arXiv:2304.13705](https://arxiv.org/abs/2304.13705)（ACT の原著論文）
- データセット: [lerobot/aloha_sim_transfer_cube_human](https://huggingface.co/datasets/lerobot/aloha_sim_transfer_cube_human)（MIT License）
- シミュレータ: [huggingface/gym-aloha](https://github.com/huggingface/gym-aloha)（Apache License 2.0）
- 到達目標の参考値: [lerobot/act_aloha_sim_transfer_cube_human](https://huggingface.co/lerobot/act_aloha_sim_transfer_cube_human)

## ライセンス

未定
