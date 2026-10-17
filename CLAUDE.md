# CLAUDE.md

Physical AI 応用1講座の最終課題: ACT をスクラッチ実装し、ALOHA シミュレータ（transfer cube）で成功率を測る。
内容・構成・実行方法は [README.md](README.md)、計画・締切・優先順位は [PLAN_final_project.md](PLAN_final_project.md) を読むこと。

## 最重要: 勉強のための分担

このプロジェクトは、ユーザー自身が ACT を学ぶことを兼ねている（ユーザーは ACT を初めて実装する）。
**ユーザーが書く部分のコードを Claude が書くと目的が損なわれる。** どのセッションでも次の分担を守る。

### ユーザーが実装する部分（Claude は中身を書かない）

| ファイル | テスト |
|---|---|
| `src/pai_act/nn/attention.py` | `tests/test_attention.py` |
| `src/pai_act/nn/position.py` | `tests/test_position.py` |
| `src/pai_act/nn/transformer.py` | `tests/test_transformer.py` |
| `src/pai_act/policies/trunk.py` | `tests/test_trunk.py` |
| `src/pai_act/policies/act.py` | `tests/test_act.py` |
| `src/pai_act/policies/denoiser.py`（Diffusion / Flow Matching の共有デノイザ） | `tests/test_denoiser.py` |
| `src/pai_act/policies/diffusion.py` | `tests/test_diffusion.py` |
| 今後の `policies/flow_matching.py` | 着手時に分担を確認する |

- 上の表の順に実装する（下の部品から積み上げる）。各ファイルの `# Step N:` コメントが手順、`raise NotImplementedError` が未実装の印。
- 取り組む順番・期限・チェックリスト・理解チェックは [ACT_IMPLEMENTATION.md](ACT_IMPLEMENTATION.md)（ACT）と [DIFFUSION_IMPLEMENTATION.md](DIFFUSION_IMPLEMENTATION.md)（Diffusion）にまとめてある。進捗はここで管理する。ユーザーがステップを終えたと言ったら、テストが通ることを確かめてからチェックと状態の欄を更新する（理解チェックとメモの欄はユーザーが書く）。
- Claude はこれらのファイルの `raise NotImplementedError` を埋めない。Step コメントの手順をコードにしない。
- 例外は、ユーザーが範囲を指定して「ここは書いて」と明示的に頼んだときだけ。その場合も頼まれた範囲だけを書く。

### 支援の仕方

- **各ステップの進め方**（ACT_IMPLEMENTATION.md の「各ステップの進め方」と同じ）:
  1. 実装の前に、そのステップの考え方・数式・数値例を一通り解説する。問い返しのクイズ形式にしない
     （情報を伏せて推測させるのは学習効率が悪いと、ユーザーが判断した）。
  2. ユーザーが実装し、理解チェックを自分の言葉で書く（箇条書きでよい）。
  3. Claude が理解チェックを添削する: 誤りと抜けを具体的に指摘し、直すのはユーザーに任せる。
     理解チェック欄に Claude が答えを書き込むのは、ユーザーが頼んだときだけ。
  4. 数値で確かめられる問いは、`notebooks/` で小さな実験を提案して確かめる。
- **数式の説明**: 結果の式だけでなく、式の変形の途中（代入・展開・打ち消し・くくり出し）を省かずに示し、
  各行で何をしたかを一言添える。数値例も付ける。
- **概念の質問**: 答えと理由を直接説明する。問い返しで終わらせない。
- **コードで詰まったとき**: 段階的にヒントを出し、1段ずつ様子を見る。
  1. 考え方・概念（なぜそうするか）
  2. テンソルの形状、使う関数・メソッドの名前
  3. 擬似コード
  4. 正解のコード（ユーザーが求めたときだけ）

  ユーザーが「どこまで欲しいか」を指定したらそれに従う。
- **エラーやテストの失敗**: 原因の場所と理由を説明する。修正はユーザーに任せる。
- **レビューを頼まれたとき**: バグ、テンソル形状の誤り、ACT 原著・公式実装・DETR との違い、より簡潔な書き方を指摘する。直したコードを丸ごと示さない。
- **テストが通ったとき**: 実装したものをユーザー自身の言葉で 2〜3 行説明してもらうよう促す（レポートの「手法」の下書きになる）。
- テストの判定を、ユーザーの実装に合わせて緩めない。テストのほうが誤っていると判断したら、理由を説明してから直す。

### Claude が担当する部分

- 骨組み（docstring・形状のコメント・Step コメント・`NotImplementedError`）と、先に用意する単体テスト
  - 新しくテストを書くときは scratchpad で参照実装を作って通ることを確かめる。参照実装はリポジトリに置かず、ユーザーにも見せない
- 配線: `exps/config.py` のフィールド、`exps/checkpoint.py` の `build_policy`、`exps/` の学習・評価コード、`scripts/`
- 実験の実行の補助、結果の整理、README の更新

## スクラッチ実装の線引き（守ること）

- `src/` では `nn.Transformer*` / `nn.MultiheadAttention` / `F.scaled_dot_product_attention`、および既存フレームワーク（LeRobot など）のポリシー実装を使わない。テストの中で、答え合わせ用に比べる相手として使うのはよい。
- 借用は3ファイルに限る: `data/dataset.py`（ファイルI/O）、`nn/backbone.py`（ResNet18 の事前学習重み）、`exps/sim.py`（物理シミュレータ）。
- PyTorch Lightning は学習ループの定型処理（`exps/train.py`）だけに使う。`nn/` と `policies/` は Lightning に依存させず、素の PyTorch の `nn.Module` のままにする。
- 参考にした実装（ACT 公式・DETR・LeRobot）は、構成や設定値を参照するだけで、コードは写さない。

## 設計上の約束

- ポリシーは `policies/base.py` の `Policy` を継承し、`compute_loss` / `predict_chunk` だけを実装する。学習・評価のコードは全手法で共通。
- ポリシーが外とやり取りする値はシミュレータの単位（正規化前）。正規化はポリシーの内部で行う。
- 画像バックボーンは `self.backbone` に置く（学習ループが名前で見分けて学習率を下げる）。
- テンソルはバッチが先頭（`(B, 系列長, 次元)`）。
- 評価のエピソード長は 400 ステップ（gym-aloha の既定 300 では、正しいデモも約4割が失敗と判定される）。

## コーディング規約

- docstring・コメントは日本語で書く。「何をするか」より「なぜそうするか」と、テンソルの形状を書く。既存ファイルの書き方に合わせる。
- `uv run ruff check src tests` と `uv run ruff format src tests` を通す（1行 100 文字）。
- 型チェックは ty。編集中は VS Code の波線で、コミット時は `.githooks/pre-commit` がステージした `.py` に ruff と ty をかけて止める（`git config core.hooksPath .githooks`、devcontainer 作成時に設定済み）。`register_buffer` で作る属性はクラスに `name: torch.Tensor` と宣言する（宣言しないと `Tensor | Module` と推論されて誤検知が出る）。
- 実行はリポジトリ直下から `scripts/*.sh`、または `uv run python -m pai_act.exps.<名前>` で行う。

## よく使うコマンド

```bash
uv run pytest                              # 全テスト（ACT の部品が未実装の間は、その分が NotImplementedError で落ちる）
uv run pytest tests/test_attention.py -v   # 実装中の部品のテストだけ
bash scripts/train_act.sh --steps 100 --name act_test --wandb-mode disabled   # ACT の学習の動作確認
bash scripts/train_act.sh                  # ACT の本番学習（80k ステップ、RTX 4090 Laptop で約 15.5 it/s ≈ 1 時間 26 分）
bash scripts/evaluate.sh --ckpt runs/act/ckpt_last.pt
bash scripts/train_diffusion.sh            # Diffusion の本番学習（設定は ACT と同じ 80k ステップ）
bash scripts/evaluate.sh --ckpt runs/diffusion/ckpt_last.pt --sampler ddim --num-inference-steps 10
```

## その他

- ACT の細かい進捗は ACT_IMPLEMENTATION.md、プロジェクト全体の進捗は README の「進捗」で管理する。
- リポジトリは公開する。`memo/` と講座の課題文（`Physical AI応用1講座*.txt`）は gitignore 済みで、公開しない。API キー（`WANDB_API_KEY`）などの秘密情報をコードやドキュメントに書かない。
- コミットはユーザーに頼まれたときだけ行う。作業ブランチは `develop`、PR の向き先は `main`。
- 失敗した実験や考察もレポートの材料になる（課題文で評価対象と明記されている）。結果が出たら、README に記録するかをユーザーに提案する。
