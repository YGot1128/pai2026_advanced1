"""(実行) 条件ごとの評価結果（results.json）を集め、比較表とグラフを作る。

使い方（リポジトリ直下で実行）:
    bash scripts/compare.sh --preset act_ablation        # 決めておいた条件の組（プリセット）で作る
    bash scripts/compare.sh --list                       # プリセットの一覧と、結果が揃っているか
    bash scripts/compare.sh --run "ACT=runs/act/eval_ckpt_last" \\
        --run "Diffusion=runs/diffusion/eval_ckpt_last_ddim10" --out-dir outputs/compare/custom
    （中身は uv run python -m pai_act.visualize.compare ...）

出力（既定は outputs/compare/<プリセット名>/。outputs/ は .gitignore 済み）:
    table.md              … 比較表。README やレポートに貼る
    table.csv             … 同じ数値を CSV で（表計算ソフトやレポートの表の作成に使う）
    success_rate.png      … 成功率と 95% 信頼区間
    outcomes.png          … 50 本の内訳（どの段階まで進んで止まったか）
    speed_vs_success.png  … predict_chunk 1回の時間と成功率（50Hz 制御の 20ms の線つき）

比較の考え方（1条件につき学習1回・評価 50 本なので、差が偶然かどうかを数字で示す）:
    - 成功率の 95% 信頼区間は Wilson の方法で出す。成功率が 0% や 100% に近くても区間が [0, 1] からはみ出さない
    - 表の最初の条件を「基準」にして、ほかの条件と比べる。検定は2通り出す
        Fisher の正確検定: 成功の本数だけで比べる（README の ACT と baseline の比較で使ったもの）
        McNemar の正確検定: 同じシード（= 同じキューブの初期位置）どうしを組にして、
            「基準だけ成功」と「この条件だけ成功」の本数を比べる。両方成功・両方失敗のシードは差に関係しないので除く。
            プリセットの中の全条件を同じシードの範囲（既定は 0〜49）で評価しているので、組にした比較ができる
    - 評価の本数を増やした条件は、評価の出力先を「+」でつなぐ（例: runs/act/eval_ckpt_last+runs/act/eval_ckpt_last_s50n150
      でシード 0〜199 の 200 本）。50 本では 95% 信頼区間の幅が ±12 pt ほどあり、手法どうしの差を区別できないため。
      つないだ出力先が1つでも欠けていれば、その条件は表から外す（50 本だけの結果を 200 本として扱わないように）
    - ただし同じチェックポイントを評価し直すだけでも成否が変わるシードがある（README の「評価の再現性」）ので、
      p 値が小さくても、それは評価1回分のばらつきを含んだうえでの判断になる

図の文字は英語にしている。コンテナに日本語フォントが入っておらず、日本語は文字化けするため。
条件の名前（ラベル）も図と表で揃えるため英語にしている。

色は dataviz のリファレンスの配色に揃えた（白に近い背景、黒に近い文字、青の1色）。
outcomes.png の 5 段階は順序のある量なので、同じ青の明→暗で塗り、進んだ段階ほど濃くする。
色だけで見分けなくて済むよう、凡例を付け、各区間に本数を書く。
"""

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 画面のない環境（コンテナ・nohup）でも PNG を書き出せるようにする

import matplotlib.pyplot as plt  # noqa: E402  （use("Agg") の後に読み込む必要がある）
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

# --- 条件の組（プリセット） ---------------------------------------------------------------
# 各プリセットは（説明, [(ラベル, 評価の出力先), ...]）。最初の条件が比較の基準になる。
# 評価の出力先の名前は exps/evaluate.py が付けるもの（Diffusion はサンプラーとステップ数、
# ensembling なしは _noens<n> が付く）。結果がまだない条件は、表とグラフから外して作る
PRESETS: dict[str, tuple[str, list[tuple[str, str]]]] = {
    "act_ablation": (
        "ACT の追加実験（ACT_IMPLEMENTATION.md の 9・10）: Decoder 層数と行動クエリの位置ベクトル",
        [
            ("ACT (dec 1, learned query)", "runs/act/eval_ckpt_last"),
            ("ACT (dec 1, learned query, seed 1)", "runs/act_seed1/eval_ckpt_last"),
            ("ACT dec 7", "runs/act_dec7/eval_ckpt_last"),
            ("ACT dec 7 (seed 1)", "runs/act_dec7_seed1/eval_ckpt_last"),
            ("ACT fixed query", "runs/act_fixedq/eval_ckpt_last"),
            ("ACT fixed query (seed 1)", "runs/act_fixedq_seed1/eval_ckpt_last"),
        ],
    ),
    "act_ensemble": (
        "ACT の temporal ensembling の重みの係数 m と有無（学習は1回で、評価だけを変える）。"
        " 重み exp(-m*i) の i = 0 が最も古い予測なので、m が大きいほど古い予測に寄る",
        [
            ("m = 0.01 (default)", "runs/act/eval_ckpt_last"),
            ("m = 0 (uniform)", "runs/act/eval_ckpt_last_coef0"),
            ("m = 0.1", "runs/act/eval_ckpt_last_coef0.1"),
            ("m = 1", "runs/act/eval_ckpt_last_coef1"),
            ("no ensembling (every 100)", "runs/act/eval_ckpt_last_noens100"),
            ("no ensembling (every 50)", "runs/act/eval_ckpt_last_noens50"),
        ],
    ),
    "heads": (
        "アクションヘッドの比較（同じ trunk・データ・評価条件）",
        [
            ("ACT (CVAE)", "runs/act/eval_ckpt_last"),
            ("Baseline (MLP)", "runs/baseline/eval_ckpt_last"),
            ("Diffusion (DDIM 10)", "runs/diffusion/eval_ckpt_last_ddim10"),
            ("Flow Matching (Euler 10)", "runs/flow_matching/eval_ckpt_last_euler10"),
        ],
    ),
    "flow_matching_solvers": (
        "Flow Matching の ODE ソルバーとステップ数（FLOW_MATCHING_IMPLEMENTATION.md の F7）。"
        " 学習は1回で、評価だけを変える。Heun は1ステップにデノイザを2回呼ぶので、NFE は Euler N = Heun N/2 で揃う",
        [
            ("Euler 1", "runs/flow_matching/eval_ckpt_last_euler1"),
            ("Euler 2", "runs/flow_matching/eval_ckpt_last_euler2"),
            ("Euler 5", "runs/flow_matching/eval_ckpt_last_euler5"),
            ("Euler 10", "runs/flow_matching/eval_ckpt_last_euler10"),
            ("Euler 20", "runs/flow_matching/eval_ckpt_last_euler20"),
            ("Heun 5", "runs/flow_matching/eval_ckpt_last_heun5"),
            ("Heun 10", "runs/flow_matching/eval_ckpt_last_heun10"),
            # 学習シード 1 でも、少ないステップ数で足りるかを確かめる（10/23 の夜間実行）
            ("Euler 1 (seed 1)", "runs/flow_matching_seed1/eval_ckpt_last_euler1"),
            ("Euler 2 (seed 1)", "runs/flow_matching_seed1/eval_ckpt_last_euler2"),
            ("Euler 5 (seed 1)", "runs/flow_matching_seed1/eval_ckpt_last_euler5"),
            ("Euler 10 (seed 1)", "runs/flow_matching_seed1/eval_ckpt_last_euler10"),
            ("Heun 5 (seed 1)", "runs/flow_matching_seed1/eval_ckpt_last_heun5"),
            # 学習シード 2 は Euler 1 だけ（10/25 の夜間実行。学習外の配置の Euler 1 と比べるため）
            ("Euler 1 (seed 2)", "runs/flow_matching_seed2/eval_ckpt_last_euler1"),
            ("Euler 10 (seed 2)", "runs/flow_matching_seed2/eval_ckpt_last_euler10"),
        ],
    ),
    "heads_seeds": (
        "アクションヘッドの比較を、学習シード 0 / 1 / 2 と temporal ensembling の有無で行う"
        "（Diffusion は DDPM 100、Flow Matching は Euler 10。ensembling なしは 100 ステップごとに推論）",
        [
            ("ACT (seed 0) + ens", "runs/act/eval_ckpt_last"),
            ("ACT (seed 1) + ens", "runs/act_seed1/eval_ckpt_last"),
            ("ACT (seed 2) + ens", "runs/act_seed2/eval_ckpt_last"),
            ("ACT (seed 0) no ens", "runs/act/eval_ckpt_last_noens100"),
            ("ACT (seed 1) no ens", "runs/act_seed1/eval_ckpt_last_noens100"),
            ("ACT (seed 2) no ens", "runs/act_seed2/eval_ckpt_last_noens100"),
            ("Diffusion DDPM 100 (seed 0) + ens", "runs/diffusion/eval_ckpt_last_ddpm100"),
            ("Diffusion DDPM 100 (seed 1) + ens", "runs/diffusion_seed1/eval_ckpt_last_ddpm100"),
            ("Diffusion DDPM 100 (seed 2) + ens", "runs/diffusion_seed2/eval_ckpt_last_ddpm100"),
            (
                "Diffusion DDPM 100 (seed 0) no ens",
                "runs/diffusion/eval_ckpt_last_ddpm100_noens100",
            ),
            (
                "Diffusion DDPM 100 (seed 1) no ens",
                "runs/diffusion_seed1/eval_ckpt_last_ddpm100_noens100",
            ),
            (
                "Diffusion DDPM 100 (seed 2) no ens",
                "runs/diffusion_seed2/eval_ckpt_last_ddpm100_noens100",
            ),
            ("Flow Matching Euler 10 (seed 0) + ens", "runs/flow_matching/eval_ckpt_last_euler10"),
            (
                "Flow Matching Euler 10 (seed 1) + ens",
                "runs/flow_matching_seed1/eval_ckpt_last_euler10",
            ),
            (
                "Flow Matching Euler 10 (seed 2) + ens",
                "runs/flow_matching_seed2/eval_ckpt_last_euler10",
            ),
            (
                "Flow Matching Euler 10 (seed 0) no ens",
                "runs/flow_matching/eval_ckpt_last_euler10_noens100",
            ),
            (
                "Flow Matching Euler 10 (seed 1) no ens",
                "runs/flow_matching_seed1/eval_ckpt_last_euler10_noens100",
            ),
            (
                "Flow Matching Euler 10 (seed 2) no ens",
                "runs/flow_matching_seed2/eval_ckpt_last_euler10_noens100",
            ),
        ],
    ),
    "eval200": (
        "評価の本数を 200 本（シード 0〜199）に増やした比較。50 本の評価（シード 0〜49）と、"
        "追加の 150 本（シード 50〜199）を合わせて集計する。シード 0〜49 は学習デモと同じ配置、"
        "50〜199 は学習外の配置なので、汎化を論じるときはプリセット ood150 で分けて見る",
        [
            (
                "ACT (seed 0) + ens",
                "runs/act/eval_ckpt_last+runs/act/eval_ckpt_last_s50n150",
            ),
            (
                "ACT (seed 1) + ens",
                "runs/act_seed1/eval_ckpt_last+runs/act_seed1/eval_ckpt_last_s50n150",
            ),
            (
                "ACT (seed 0) no ens",
                "runs/act/eval_ckpt_last_noens100+runs/act/eval_ckpt_last_noens100_s50n150",
            ),
            (
                "ACT (seed 1) no ens",
                "runs/act_seed1/eval_ckpt_last_noens100"
                "+runs/act_seed1/eval_ckpt_last_noens100_s50n150",
            ),
            (
                "ACT fixed query (seed 0) + ens",
                "runs/act_fixedq/eval_ckpt_last+runs/act_fixedq/eval_ckpt_last_s50n150",
            ),
            (
                "ACT fixed query (seed 1) + ens",
                "runs/act_fixedq_seed1/eval_ckpt_last+runs/act_fixedq_seed1/eval_ckpt_last_s50n150",
            ),
            (
                "ACT fixed query (seed 0) no ens",
                "runs/act_fixedq/eval_ckpt_last_noens100"
                "+runs/act_fixedq/eval_ckpt_last_noens100_s50n150",
            ),
            (
                "ACT fixed query (seed 1) no ens",
                "runs/act_fixedq_seed1/eval_ckpt_last_noens100"
                "+runs/act_fixedq_seed1/eval_ckpt_last_noens100_s50n150",
            ),
            (
                "Flow Matching Euler 10 (seed 0) + ens",
                "runs/flow_matching/eval_ckpt_last_euler10"
                "+runs/flow_matching/eval_ckpt_last_euler10_s50n150",
            ),
            (
                "Flow Matching Euler 10 (seed 1) + ens",
                "runs/flow_matching_seed1/eval_ckpt_last_euler10"
                "+runs/flow_matching_seed1/eval_ckpt_last_euler10_s50n150",
            ),
            (
                "Flow Matching Euler 10 (seed 0) no ens",
                "runs/flow_matching/eval_ckpt_last_euler10_noens100"
                "+runs/flow_matching/eval_ckpt_last_euler10_noens100_s50n150",
            ),
            (
                "Flow Matching Euler 10 (seed 1) no ens",
                "runs/flow_matching_seed1/eval_ckpt_last_euler10_noens100"
                "+runs/flow_matching_seed1/eval_ckpt_last_euler10_noens100_s50n150",
            ),
            # Diffusion の DDPM 100 は ensembling ありの 150 本に約 2 時間かかるので、10/24 の夜間実行で追加した
            (
                "Diffusion DDPM 100 (seed 0) + ens",
                "runs/diffusion/eval_ckpt_last_ddpm100+runs/diffusion/eval_ckpt_last_ddpm100_s50n150",
            ),
            (
                "Diffusion DDPM 100 (seed 1) + ens",
                "runs/diffusion_seed1/eval_ckpt_last_ddpm100"
                "+runs/diffusion_seed1/eval_ckpt_last_ddpm100_s50n150",
            ),
            (
                "Diffusion DDPM 100 (seed 0) no ens",
                "runs/diffusion/eval_ckpt_last_ddpm100_noens100"
                "+runs/diffusion/eval_ckpt_last_ddpm100_noens100_s50n150",
            ),
            (
                "Diffusion DDPM 100 (seed 1) no ens",
                "runs/diffusion_seed1/eval_ckpt_last_ddpm100_noens100"
                "+runs/diffusion_seed1/eval_ckpt_last_ddpm100_noens100_s50n150",
            ),
        ],
    ),
    "ood150": (
        "学習外の配置（シード 50〜199 の 150 本）だけでの比較。シード 0〜49 は学習デモと同じ配置なので含めない"
        "（docs/EVALUATION.md）。3 手法と ACT の行動クエリ固定版を、学習シード 0 / 1 / 2 と temporal ensembling の有無で比べる",
        [
            ("ACT (seed 0) + ens", "runs/act/eval_ckpt_last_s50n150"),
            ("ACT (seed 1) + ens", "runs/act_seed1/eval_ckpt_last_s50n150"),
            ("ACT (seed 2) + ens", "runs/act_seed2/eval_ckpt_last_s50n150"),
            ("ACT (seed 0) no ens", "runs/act/eval_ckpt_last_noens100_s50n150"),
            ("ACT (seed 1) no ens", "runs/act_seed1/eval_ckpt_last_noens100_s50n150"),
            ("ACT (seed 2) no ens", "runs/act_seed2/eval_ckpt_last_noens100_s50n150"),
            ("ACT fixed query (seed 0) + ens", "runs/act_fixedq/eval_ckpt_last_s50n150"),
            ("ACT fixed query (seed 1) + ens", "runs/act_fixedq_seed1/eval_ckpt_last_s50n150"),
            ("ACT fixed query (seed 2) + ens", "runs/act_fixedq_seed2/eval_ckpt_last_s50n150"),
            ("ACT fixed query (seed 0) no ens", "runs/act_fixedq/eval_ckpt_last_noens100_s50n150"),
            (
                "ACT fixed query (seed 1) no ens",
                "runs/act_fixedq_seed1/eval_ckpt_last_noens100_s50n150",
            ),
            (
                "ACT fixed query (seed 2) no ens",
                "runs/act_fixedq_seed2/eval_ckpt_last_noens100_s50n150",
            ),
            ("Diffusion DDPM 100 (seed 0) + ens", "runs/diffusion/eval_ckpt_last_ddpm100_s50n150"),
            (
                "Diffusion DDPM 100 (seed 1) + ens",
                "runs/diffusion_seed1/eval_ckpt_last_ddpm100_s50n150",
            ),
            (
                "Diffusion DDPM 100 (seed 2) + ens",
                "runs/diffusion_seed2/eval_ckpt_last_ddpm100_s50n150",
            ),
            (
                "Diffusion DDPM 100 (seed 0) no ens",
                "runs/diffusion/eval_ckpt_last_ddpm100_noens100_s50n150",
            ),
            (
                "Diffusion DDPM 100 (seed 1) no ens",
                "runs/diffusion_seed1/eval_ckpt_last_ddpm100_noens100_s50n150",
            ),
            (
                "Diffusion DDPM 100 (seed 2) no ens",
                "runs/diffusion_seed2/eval_ckpt_last_ddpm100_noens100_s50n150",
            ),
            (
                "Flow Matching Euler 10 (seed 0) + ens",
                "runs/flow_matching/eval_ckpt_last_euler10_s50n150",
            ),
            (
                "Flow Matching Euler 10 (seed 1) + ens",
                "runs/flow_matching_seed1/eval_ckpt_last_euler10_s50n150",
            ),
            (
                "Flow Matching Euler 10 (seed 2) + ens",
                "runs/flow_matching_seed2/eval_ckpt_last_euler10_s50n150",
            ),
            (
                "Flow Matching Euler 10 (seed 0) no ens",
                "runs/flow_matching/eval_ckpt_last_euler10_noens100_s50n150",
            ),
            (
                "Flow Matching Euler 10 (seed 1) no ens",
                "runs/flow_matching_seed1/eval_ckpt_last_euler10_noens100_s50n150",
            ),
            (
                "Flow Matching Euler 10 (seed 2) no ens",
                "runs/flow_matching_seed2/eval_ckpt_last_euler10_noens100_s50n150",
            ),
        ],
    ),
    # 途中のチェックポイントの 3 プリセットは、どれも最後（80k = ckpt_last）を基準にして、
    # 「途中で止めても成功率は同じか」を McNemar で見る。手法ごとに分けるのは、基準を手法の中でそろえるため
    "ckpt_progress_act": (
        "ACT の途中のチェックポイント（10k〜80k ステップ）の成功率（学習シード 0）。"
        " 検証 L1 は 30k から横ばいになるが、成功率も同じ時期に頭打ちになるかを見る。80k は ckpt_last",
        [
            ("ACT 80k", "runs/act/eval_ckpt_last"),
            *[(f"ACT {k}0k", f"runs/act/eval_ckpt_0{k}0000") for k in range(1, 8)],
        ],
    ),
    "ckpt_progress_flow_matching": (
        "Flow Matching（Euler 10）の途中のチェックポイント（10k〜80k ステップ）の成功率（学習シード 0）。"
        " 検証 MSE は 16k で最小になり、その後は上がり続けたが、成功率も下がるかを見る。80k は ckpt_last",
        [
            ("Flow Matching Euler 10 80k", "runs/flow_matching/eval_ckpt_last_euler10"),
            *[
                (f"Flow Matching Euler 10 {k}0k", f"runs/flow_matching/eval_ckpt_0{k}0000_euler10")
                for k in range(1, 8)
            ],
        ],
    ),
    "ckpt_progress_diffusion": (
        "Diffusion（DDPM 100）の途中のチェックポイントの成功率（学習シード 0）。検証 MSE は 18k で最小になり、"
        "その後は上下しながら上がった。DDPM 100 は 1 本約 40 分なので 20k と 40k だけ。80k は ckpt_last",
        [
            ("Diffusion DDPM 100 80k", "runs/diffusion/eval_ckpt_last_ddpm100"),
            ("Diffusion DDPM 100 20k", "runs/diffusion/eval_ckpt_020000_ddpm100"),
            ("Diffusion DDPM 100 40k", "runs/diffusion/eval_ckpt_040000_ddpm100"),
        ],
    ),
    # 学習外の配置（シード 50〜199）での途中のチェックポイント。検証損失は学習外の 5 本で測るので、
    # 学習外の配置の成功率なら検証損失と同じ向きに動くかを見る。20k・40k だけで、80k は ckpt_last の学習外の評価
    "ckpt_progress_ood_act": (
        "ACT の途中のチェックポイントの成功率を、学習外の配置（評価シード 50〜199 の 150 本）で見る（学習シード 0）",
        [
            ("ACT 80k (seeds 50-199)", "runs/act/eval_ckpt_last_s50n150"),
            ("ACT 20k (seeds 50-199)", "runs/act/eval_ckpt_020000_s50n150"),
            ("ACT 40k (seeds 50-199)", "runs/act/eval_ckpt_040000_s50n150"),
        ],
    ),
    "ckpt_progress_ood_flow_matching": (
        "Flow Matching（Euler 10）の途中のチェックポイントの成功率を、学習外の配置（評価シード 50〜199 の 150 本）で見る"
        "（学習シード 0）",
        [
            (
                "Flow Matching Euler 10 80k (seeds 50-199)",
                "runs/flow_matching/eval_ckpt_last_euler10_s50n150",
            ),
            (
                "Flow Matching Euler 10 20k (seeds 50-199)",
                "runs/flow_matching/eval_ckpt_020000_euler10_s50n150",
            ),
            (
                "Flow Matching Euler 10 40k (seeds 50-199)",
                "runs/flow_matching/eval_ckpt_040000_euler10_s50n150",
            ),
        ],
    ),
    "ckpt_progress_ood_diffusion": (
        "Diffusion（DDPM 100）の途中のチェックポイントの成功率を、学習外の配置（評価シード 50〜199 の 150 本）で見る"
        "（学習シード 0）。学習内の配置では 20k が 0%、40k が 46%、80k が 72% だった",
        [
            (
                "Diffusion DDPM 100 80k (seeds 50-199)",
                "runs/diffusion/eval_ckpt_last_ddpm100_s50n150",
            ),
            (
                "Diffusion DDPM 100 20k (seeds 50-199)",
                "runs/diffusion/eval_ckpt_020000_ddpm100_s50n150",
            ),
            (
                "Diffusion DDPM 100 40k (seeds 50-199)",
                "runs/diffusion/eval_ckpt_040000_ddpm100_s50n150",
            ),
        ],
    ),
    # 学習シード 1 の途中のチェックポイント（学習内の配置）。学習シード 0 の ckpt_progress_* と同じ傾向が出るかを見る
    "ckpt_progress_act_seed1": (
        "ACT の途中のチェックポイント（10k〜80k ステップ）の成功率（学習シード 1）。80k は ckpt_last",
        [
            ("ACT seed 1 80k", "runs/act_seed1/eval_ckpt_last"),
            *[(f"ACT seed 1 {k}0k", f"runs/act_seed1/eval_ckpt_0{k}0000") for k in range(1, 8)],
        ],
    ),
    "ckpt_progress_flow_matching_seed1": (
        "Flow Matching（Euler 10）の途中のチェックポイント（10k〜80k ステップ）の成功率（学習シード 1）。"
        " 検証 MSE は 13k で最小になった。80k は ckpt_last",
        [
            (
                "Flow Matching Euler 10 seed 1 80k",
                "runs/flow_matching_seed1/eval_ckpt_last_euler10",
            ),
            *[
                (
                    f"Flow Matching Euler 10 seed 1 {k}0k",
                    f"runs/flow_matching_seed1/eval_ckpt_0{k}0000_euler10",
                )
                for k in range(1, 8)
            ],
        ],
    ),
    "ckpt_progress_diffusion_seed1": (
        "Diffusion（DDPM 100）の途中のチェックポイントの成功率（学習シード 1）。学習シード 0 では 20k が 0% だったので、"
        "立ち上がりの遅さが学習シードによらないかを見る。80k は ckpt_last",
        [
            ("Diffusion DDPM 100 seed 1 80k", "runs/diffusion_seed1/eval_ckpt_last_ddpm100"),
            ("Diffusion DDPM 100 seed 1 20k", "runs/diffusion_seed1/eval_ckpt_020000_ddpm100"),
            ("Diffusion DDPM 100 seed 1 40k", "runs/diffusion_seed1/eval_ckpt_040000_ddpm100"),
        ],
    ),
    "ckpt_progress_diffusion_seed2": (
        "Diffusion（DDPM 100）の途中のチェックポイントの成功率（学習シード 2）。学習シード 0 / 1 と同じく、"
        "立ち上がりが遅いかを見る。80k は ckpt_last",
        [
            ("Diffusion DDPM 100 seed 2 80k", "runs/diffusion_seed2/eval_ckpt_last_ddpm100"),
            ("Diffusion DDPM 100 seed 2 20k", "runs/diffusion_seed2/eval_ckpt_020000_ddpm100"),
            ("Diffusion DDPM 100 seed 2 40k", "runs/diffusion_seed2/eval_ckpt_040000_ddpm100"),
        ],
    ),
    # 学習外の配置（シード 50〜199 の 150 本）で、学習内の配置だけで比べていた設定を確かめる 2 プリセット
    "flow_matching_euler1_ood": (
        "Flow Matching の Euler 1（ACT と同じ速さ）と Euler 10 を、学習外の配置（評価シード 50〜199 の 150 本）で比べる"
        "（学習シード 0 / 1 / 2、ensembling あり）。学習内の配置では、どちらも 76〜86% だった（F7）",
        [
            *[
                (f"Flow Matching Euler 10 (seed {i})", f"runs/{run}/eval_ckpt_last_euler10_s50n150")
                for i, run in enumerate(
                    ["flow_matching", "flow_matching_seed1", "flow_matching_seed2"]
                )
            ],
            *[
                (f"Flow Matching Euler 1 (seed {i})", f"runs/{run}/eval_ckpt_last_euler1_s50n150")
                for i, run in enumerate(
                    ["flow_matching", "flow_matching_seed1", "flow_matching_seed2"]
                )
            ],
        ],
    ),
    "act_dec7_ood": (
        "ACT の Decoder 1 層（既定）と 7 層を、学習外の配置（評価シード 50〜199 の 150 本）で比べる"
        "（学習シード 0 / 1、ensembling あり）。学習内の配置では差がなかった（ACT_IMPLEMENTATION.md のステップ 9）",
        [
            ("ACT dec 1 (seed 0)", "runs/act/eval_ckpt_last_s50n150"),
            ("ACT dec 1 (seed 1)", "runs/act_seed1/eval_ckpt_last_s50n150"),
            ("ACT dec 7 (seed 0)", "runs/act_dec7/eval_ckpt_last_s50n150"),
            ("ACT dec 7 (seed 1)", "runs/act_dec7_seed1/eval_ckpt_last_s50n150"),
        ],
    ),
    "eval_repeat": (
        "同じチェックポイントを同じ条件（評価シード 0〜49）で評価し直したときのばらつき（学習シード 0）。"
        " 学習は同じなので、差は評価だけの揺れ（GPU の丸め誤差と、Diffusion / Flow Matching では推論の乱数）",
        [
            ("ACT (original)", "runs/act/eval_ckpt_last"),
            *[(f"ACT (rep {i})", f"runs/act/eval_ckpt_last_rep{i}") for i in range(1, 4)],
            ("Flow Matching Euler 10 (original)", "runs/flow_matching/eval_ckpt_last_euler10"),
            *[
                (
                    f"Flow Matching Euler 10 (rep {i})",
                    f"runs/flow_matching/eval_ckpt_last_euler10_rep{i}",
                )
                for i in range(1, 4)
            ],
            ("Diffusion DDPM 100 (original)", "runs/diffusion/eval_ckpt_last_ddpm100"),
            *[
                (f"Diffusion DDPM 100 (rep {i})", f"runs/diffusion/eval_ckpt_last_ddpm100_rep{i}")
                for i in range(1, 3)
            ],
        ],
    ),
    "act_noens": (
        "ACT の temporal ensembling の有無を、学習シード・設定違いのチェックポイントで比べる"
        "（ACT_IMPLEMENTATION.md のステップ 11。シード 0 の 1 層だけで 72% → 84% だった）",
        [
            ("ACT (seed 0) + ens", "runs/act/eval_ckpt_last"),
            ("ACT (seed 0) no ens", "runs/act/eval_ckpt_last_noens100"),
            ("ACT (seed 1) + ens", "runs/act_seed1/eval_ckpt_last"),
            ("ACT (seed 1) no ens", "runs/act_seed1/eval_ckpt_last_noens100"),
            ("ACT dec 7 (seed 0) + ens", "runs/act_dec7/eval_ckpt_last"),
            ("ACT dec 7 (seed 0) no ens", "runs/act_dec7/eval_ckpt_last_noens100"),
            ("ACT dec 7 (seed 1) + ens", "runs/act_dec7_seed1/eval_ckpt_last"),
            ("ACT dec 7 (seed 1) no ens", "runs/act_dec7_seed1/eval_ckpt_last_noens100"),
            ("ACT fixed query (seed 0) + ens", "runs/act_fixedq/eval_ckpt_last"),
            ("ACT fixed query (seed 0) no ens", "runs/act_fixedq/eval_ckpt_last_noens100"),
            ("ACT fixed query (seed 1) + ens", "runs/act_fixedq_seed1/eval_ckpt_last"),
            ("ACT fixed query (seed 1) no ens", "runs/act_fixedq_seed1/eval_ckpt_last_noens100"),
            ("ACT (seed 2) + ens", "runs/act_seed2/eval_ckpt_last"),
            ("ACT (seed 2) no ens", "runs/act_seed2/eval_ckpt_last_noens100"),
            ("ACT fixed query (seed 2) + ens", "runs/act_fixedq_seed2/eval_ckpt_last"),
            ("ACT fixed query (seed 2) no ens", "runs/act_fixedq_seed2/eval_ckpt_last_noens100"),
        ],
    ),
    "diffusion_steps": (
        "Diffusion の推論ステップ数（DIFFUSION_IMPLEMENTATION.md の D9）。学習は1回で、評価だけを変える",
        [
            ("DDIM 10 (clip 6)", "runs/diffusion/eval_ckpt_last_ddim10"),
            ("DDIM 5 (clip 6)", "runs/diffusion/eval_ckpt_last_ddim5"),
            ("DDIM 20 (clip 6)", "runs/diffusion/eval_ckpt_last_ddim20"),
            ("DDIM 100 (clip 6)", "runs/diffusion/eval_ckpt_last_ddim100"),
            ("DDIM 5 (clip 2)", "runs/diffusion/eval_ckpt_last_ddim5_clip2"),
            ("DDIM 10 (clip 2)", "runs/diffusion/eval_ckpt_last_ddim10_clip2"),
            ("DDIM 20 (clip 2)", "runs/diffusion/eval_ckpt_last_ddim20_clip2"),
            ("DDPM 100 (clip 6)", "runs/diffusion/eval_ckpt_last_ddpm100"),
            ("DDPM 100 (clip 2)", "runs/diffusion/eval_ckpt_last_ddpm100_clip2"),
        ],
    ),
    "diffusion_eta": (
        "DDIM に毎ステップ乱数を足す η（DIFFUSION_IMPLEMENTATION.md の D11）。学習は1回で、評価だけを変える。"
        " η = 1 の DDIM 100 は、x̂_0 のクリップの扱い以外は DDPM 100 と同じ更新",
        [
            ("DDIM 100 clip 6, eta 0", "runs/diffusion/eval_ckpt_last_ddim100"),
            ("DDIM 100 clip 6, eta 0.5", "runs/diffusion/eval_ckpt_last_ddim100_eta0.5"),
            ("DDIM 100 clip 6, eta 1", "runs/diffusion/eval_ckpt_last_ddim100_eta1"),
            ("DDPM 100 clip 6", "runs/diffusion/eval_ckpt_last_ddpm100"),
            ("DDIM 10 clip 2, eta 0", "runs/diffusion/eval_ckpt_last_ddim10_clip2"),
            ("DDIM 10 clip 2, eta 0.5", "runs/diffusion/eval_ckpt_last_ddim10_clip2_eta0.5"),
            ("DDIM 10 clip 2, eta 1", "runs/diffusion/eval_ckpt_last_ddim10_clip2_eta1"),
            ("DDIM 20 clip 2, eta 0", "runs/diffusion/eval_ckpt_last_ddim20_clip2"),
            ("DDIM 20 clip 2, eta 1", "runs/diffusion/eval_ckpt_last_ddim20_clip2_eta1"),
            # 学習シード 1 でも η の効果が出るかを確かめる（10/23 の夜間実行）
            ("DDIM 100 clip 6, eta 1 (seed 1)", "runs/diffusion_seed1/eval_ckpt_last_ddim100_eta1"),
            ("DDPM 100 clip 6 (seed 1)", "runs/diffusion_seed1/eval_ckpt_last_ddpm100"),
            ("DDIM 10 clip 2, eta 0 (seed 1)", "runs/diffusion_seed1/eval_ckpt_last_ddim10_clip2"),
            (
                "DDIM 10 clip 2, eta 1 (seed 1)",
                "runs/diffusion_seed1/eval_ckpt_last_ddim10_clip2_eta1",
            ),
        ],
    ),
    "diffusion_fix": (
        "Diffusion の成功率を上げる（DIFFUSION_IMPLEMENTATION.md の D8「成功率 2% の切り分け」の続き）:"
        " サンプラー・x̂_0 のクリップ・Decoder 層数",
        [
            ("dec 1, DDIM 10 (clip 6)", "runs/diffusion/eval_ckpt_last_ddim10"),
            ("dec 1, DDIM 10 (clip 2)", "runs/diffusion/eval_ckpt_last_ddim10_clip2"),
            ("dec 1, DDIM 100 (clip 6)", "runs/diffusion/eval_ckpt_last_ddim100"),
            ("dec 1, DDPM 100 (clip 6)", "runs/diffusion/eval_ckpt_last_ddpm100"),
            ("dec 4, DDIM 10 (clip 2)", "runs/diffusion_dec4/eval_ckpt_last_ddim10_clip2"),
            ("dec 4, DDPM 100 (clip 6)", "runs/diffusion_dec4/eval_ckpt_last_ddpm100"),
            ("ACT (reference)", "runs/act/eval_ckpt_last"),
        ],
    ),
    "diffusion_ensemble": (
        "Diffusion の temporal ensembling の有無（DIFFUSION_IMPLEMENTATION.md の D10）",
        [
            ("DDPM 100 + ensembling", "runs/diffusion/eval_ckpt_last_ddpm100"),
            ("DDPM 100, no ensembling (every 50)", "runs/diffusion/eval_ckpt_last_ddpm100_noens50"),
            ("DDPM 100, no ensembling (every 25)", "runs/diffusion/eval_ckpt_last_ddpm100_noens25"),
            ("DDIM 10 + ensembling", "runs/diffusion/eval_ckpt_last_ddim10"),
            ("DDIM 10, no ensembling (every 50)", "runs/diffusion/eval_ckpt_last_ddim10_noens50"),
        ],
    ),
    "seeds": (
        "学習シードによるばらつき（同じ設定で --seed だけ変える。エピソード分割・初期値・サンプル順が変わる）",
        [
            ("ACT (seed 0)", "runs/act/eval_ckpt_last"),
            ("ACT (seed 1)", "runs/act_seed1/eval_ckpt_last"),
            ("ACT dec 7 (seed 0)", "runs/act_dec7/eval_ckpt_last"),
            ("ACT dec 7 (seed 1)", "runs/act_dec7_seed1/eval_ckpt_last"),
            ("ACT fixed query (seed 0)", "runs/act_fixedq/eval_ckpt_last"),
            ("ACT fixed query (seed 1)", "runs/act_fixedq_seed1/eval_ckpt_last"),
            ("Diffusion DDPM 100 (seed 0)", "runs/diffusion/eval_ckpt_last_ddpm100"),
            ("Diffusion DDPM 100 (seed 1)", "runs/diffusion_seed1/eval_ckpt_last_ddpm100"),
            ("Diffusion DDIM 10 clip 2 (seed 0)", "runs/diffusion/eval_ckpt_last_ddim10_clip2"),
            (
                "Diffusion DDIM 10 clip 2 (seed 1)",
                "runs/diffusion_seed1/eval_ckpt_last_ddim10_clip2",
            ),
            ("Flow Matching Euler 10 (seed 0)", "runs/flow_matching/eval_ckpt_last_euler10"),
            ("Flow Matching Euler 10 (seed 1)", "runs/flow_matching_seed1/eval_ckpt_last_euler10"),
            ("ACT (seed 2)", "runs/act_seed2/eval_ckpt_last"),
            ("ACT fixed query (seed 2)", "runs/act_fixedq_seed2/eval_ckpt_last"),
            ("Diffusion DDPM 100 (seed 2)", "runs/diffusion_seed2/eval_ckpt_last_ddpm100"),
            ("Flow Matching Euler 10 (seed 2)", "runs/flow_matching_seed2/eval_ckpt_last_euler10"),
        ],
    ),
}

# 最大報酬 r で止まったエピソードの呼び名（exps/evaluate.py の FAILURE_TYPES と、成功）。
# 順番は報酬の段階の順（0: 触れない → 4: 成功）
OUTCOMES = ["no_touch", "not_lifted", "not_transferred", "transfer_failed", "success"]
STAGES = ["touched", "lifted", "transfer attempted"]  # reached の段階（max_reward ≥ 1, 2, 3）
CONTROL_PERIOD_MS = 20.0  # 50Hz 制御の1ステップ。predict_chunk がこれより遅いと実機では間に合わない

# --- 色（dataviz のリファレンス配色、ライトモード。レポートの PDF に貼るのでライトだけ） ---
SURFACE = "#fcfcfb"  # 図の背景
INK = "#0b0b0b"  # 主な文字
INK_SECONDARY = "#52514e"  # 補足の文字
INK_MUTED = "#898781"  # 軸の目盛り
GRID = "#e1e0d9"  # 補助線（目立たせない）
BASELINE = "#c3c2b7"  # 軸の線
SERIES = "#2a78d6"  # 1系列の図の色（青）
# outcomes.png の 5 段階。青の step 250 / 350 / 450 / 550 / 700。順序のある色は、最も薄い段階でも
# 背景と見分けられるよう step 250 より薄くしない（リファレンスの ordinal の約束）。
# 最後の success だけ 1 段飛ばして濃くし、隣の transfer_failed と見分けやすくする
OUTCOME_COLORS = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
# 各区間の数字の色。薄い 2 段階は黒、濃い 3 段階は白にして読めるようにする
OUTCOME_TEXT = [INK, INK, "#ffffff", "#ffffff", "#ffffff"]


@dataclass
class RunResult:
    """1条件の評価結果。results.json の古い形式（baseline の評価には failure_types などがない）でも、
    エピソードごとの max_reward から計算し直して同じ項目を揃える。"""

    label: str
    paths: list[Path]  # 評価の出力先。本数を増やした条件は複数
    seeds: np.ndarray  # (N,) リセットに使ったシード
    max_rewards: np.ndarray  # (N,) エピソード中の最大報酬 0〜4
    steps: np.ndarray  # (N,) 実行したステップ数（成功ならその時点、失敗なら最大ステップ数）
    predict_ms: float | None  # predict_chunk 1回の時間の中央値。古い results.json にはない

    @property
    def n(self) -> int:
        return len(self.max_rewards)

    @property
    def successes(self) -> int:
        return int(np.sum(self.max_rewards == 4))

    @property
    def success_rate(self) -> float:
        return self.successes / self.n

    def outcome_counts(self) -> list[int]:
        """OUTCOMES の順に、各段階で止まった本数（合計 N）。"""
        return [int(np.sum(self.max_rewards == r)) for r in range(len(OUTCOMES))]

    def reached(self) -> list[float]:
        """STAGES の順に、その段階以上まで進んだ割合。"""
        return [float(np.mean(self.max_rewards >= r)) for r in range(1, len(STAGES) + 1)]

    def success_steps(self) -> np.ndarray:
        """成功したエピソードの、成功までのステップ数。"""
        return self.steps[self.max_rewards == 4]


def eval_dirs(path: str) -> list[Path]:
    """「+」でつないだ評価の出力先を分ける（つないでいなければ1つだけのリスト）。"""
    return [Path(p) for p in path.split("+")]


def has_results(path: str) -> bool:
    """つないだ出力先のすべてに results.json があるか。"""
    return all((d / "results.json").exists() for d in eval_dirs(path))


def load_result(label: str, path: str) -> RunResult:
    """評価の出力先（「+」でつないだものは全部）から results.json を読み、エピソードを合わせて RunResult にする。"""
    dirs = eval_dirs(path)
    episodes, predict_medians = [], []
    for d in dirs:
        data = json.loads((d / "results.json").read_text())
        episodes += data["episodes"]
        if data.get("predict_ms"):
            predict_medians.append(float(data["predict_ms"]["median"]))
    seeds = np.array([e["seed"] for e in episodes])
    # 同じシードが2回入ると、組にした比較（McNemar）でどちらを使うかが決まらず、本数も水増しになる
    if len(np.unique(seeds)) != len(seeds):
        raise ValueError(f"{label}: つないだ評価の間でシードが重なっている（{path}）")
    return RunResult(
        label=label,
        paths=dirs,
        seeds=seeds,
        max_rewards=np.array([e["max_reward"] for e in episodes]),
        steps=np.array([e["steps"] for e in episodes]),
        # 評価ごとの中央値の中央値（エピソードごとの時間は残していないため。評価の間でほぼ同じ値になる）
        predict_ms=float(np.median(predict_medians)) if predict_medians else None,
    )


# --- 統計（scipy は直接の依存にしていないので、標準ライブラリの math.comb で書く） ---------------


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """成功 k 回 / n 回の成功率の 95% 信頼区間（Wilson の方法）。

    正規近似 p ± z·√(p(1−p)/n) は、p が 0 や 1 に近いと区間が [0, 1] からはみ出したり、幅が 0 になったりする。
    Wilson の方法は「真の成功率が p なら、観測値は z 標準偏差の範囲に入る」を p について解いたもので、
    区間の中心が 1/2 の側に少し寄り、端でも正しく振る舞う。
    例: 36/50 → [0.58, 0.83]（README の ACT の値）
    """
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return center - half, center + half


def fisher_exact_p(k1: int, n1: int, k2: int, n2: int) -> float:
    """2群の成功率の差の Fisher の正確検定（両側）の p 値。

    成功の合計 K = k1 + k2 を固定したとき、群 1 の成功数 x は超幾何分布 C(K, x)·C(N−K, n1−x) / C(N, n1) に従う
    （「どちらの群でも成功率が同じ」なら、K 個の成功が 2 群にでたらめに振り分けられたのと同じ）。
    観測した k1 以下の確率になる x の確率をすべて足したものが両側の p 値。
    例: ACT 36/50 と baseline 30/50 → p ≈ 0.29（README の値）
    """
    big_n, big_k = n1 + n2, k1 + k2

    def prob(x: int) -> float:
        return math.comb(big_k, x) * math.comb(big_n - big_k, n1 - x) / math.comb(big_n, n1)

    p_obs = prob(k1)
    xs = range(max(0, big_k - n2), min(big_k, n1) + 1)
    # 浮動小数点の誤差で「同じ確率」の x を取りこぼさないよう、わずかに余裕を持たせて比べる
    return min(1.0, sum(prob(x) for x in xs if prob(x) <= p_obs * (1 + 1e-7)))


def mcnemar_exact_p(b: int, c: int) -> float:
    """組にした比較の McNemar の正確検定（両側）の p 値。

    b = 基準だけ成功したシードの数、c = この条件だけ成功したシードの数。両方成功・両方失敗のシードは除く。
    「2条件の成功率が同じ」なら、成否が分かれた b + c 本は、どちらが成功するかが半々のはずなので、
    b は二項分布 Bin(b + c, 1/2) に従う。小さいほうの側の確率を 2 倍したものが両側の p 値。
    例: b = 6, c = 1 → p = 2 × (1 + 7) / 128 = 0.125
    """
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired_counts(ref: RunResult, other: RunResult) -> tuple[int, int, int] | None:
    """同じシードどうしで成否を比べ、（基準だけ成功, この条件だけ成功, 共通のシードの数）を返す。

    共通のシードがなければ None（評価のシードを変えた結果どうしは組にできない）。
    """
    ref_ok = dict(zip(ref.seeds.tolist(), (ref.max_rewards == 4).tolist(), strict=True))
    other_ok = dict(zip(other.seeds.tolist(), (other.max_rewards == 4).tolist(), strict=True))
    common = sorted(set(ref_ok) & set(other_ok))
    if not common:
        return None
    b = sum(ref_ok[s] and not other_ok[s] for s in common)
    c = sum(other_ok[s] and not ref_ok[s] for s in common)
    return b, c, len(common)


# --- 表 ------------------------------------------------------------------------------------------


def fmt_pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def fmt_steps(steps: np.ndarray) -> str:
    """成功までのステップ数を「最小 / 中央値 / 最大」で書く。成功がなければ —。"""
    if len(steps) == 0:
        return "—"
    return f"{steps.min()} / {np.median(steps):.0f} / {steps.max()}"


def write_tables(results: list[RunResult], title: str, out_dir: Path) -> None:
    """比較表を table.md（2つの表: 結果 / 基準との比較）と table.csv に書き出す。"""
    ref = results[0]
    md = [f"# {title}", ""]
    # 評価したシードの範囲を、結果から書く。学習外の配置だけの比較（シード 50〜199）もあるので、
    # 「シード 0 から」と決め打ちしない
    ranges = sorted({(int(r.seeds.min()), int(r.seeds.max()), r.n) for r in results})
    descs = [f"シード {lo}〜{hi} の {n} 本" for lo, hi, n in ranges]
    episodes_desc = descs[0] if len(descs) == 1 else f"{' / '.join(descs)}（成功率の欄の分母）"
    md.append(
        f"各条件: 学習 1 回、評価は{episodes_desc}、エピソード長 400 ステップ。"
        "失敗の型の定義は docs/EVALUATION.md。"
    )
    md += ["", "## 結果", ""]
    md.append(
        "| 条件 | 成功率 | 95% 信頼区間 | 触れた / 持ち上げた / 受け渡し試行 "
        "| 失敗の内訳（no_touch / not_lifted / not_transferred / transfer_failed） "
        "| 成功までのステップ（最小 / 中央値 / 最大） | predict_chunk（中央値） |"
    )
    md.append("|---|---|---|---|---|---|---|")
    rows = []
    for r in results:
        lo, hi = wilson_interval(r.successes, r.n)
        counts = r.outcome_counts()
        ms = "—" if r.predict_ms is None else f"{r.predict_ms:.1f} ms"
        md.append(
            f"| {r.label} | **{fmt_pct(r.success_rate)}**（{r.successes}/{r.n}） "
            f"| {fmt_pct(lo)}〜{fmt_pct(hi)} | {' / '.join(fmt_pct(x) for x in r.reached())} "
            f"| {' / '.join(str(c) for c in counts[:4])} | {fmt_steps(r.success_steps())} | {ms} |"
        )
        rows.append(
            {
                "label": r.label,
                "path": "+".join(str(p) for p in r.paths),
                "episodes": r.n,
                "successes": r.successes,
                "success_rate": round(r.success_rate, 4),
                "ci95_low": round(lo, 4),
                "ci95_high": round(hi, 4),
                **{
                    f"reached_{s.replace(' ', '_')}": x
                    for s, x in zip(STAGES, r.reached(), strict=True)
                },
                **dict(zip(OUTCOMES, counts, strict=True)),
                "predict_ms_median": None if r.predict_ms is None else round(r.predict_ms, 2),
            }
        )

    if len(results) > 1:
        md += ["", f"## 基準（{ref.label}）との比較", ""]
        md.append(
            "| 条件 | 成功率の差 | Fisher p | 成否が分かれたシード（基準だけ成功 / この条件だけ成功） "
            "| McNemar p |"
        )
        md.append("|---|---|---|---|---|")
        for r in results[1:]:
            diff = (r.success_rate - ref.success_rate) * 100
            p_fisher = fisher_exact_p(ref.successes, ref.n, r.successes, r.n)
            paired = paired_counts(ref, r)
            if paired is None:
                split, p_mcnemar = "共通のシードなし", "—"
            else:
                b, c, _ = paired
                split, p_mcnemar = f"{b} / {c}", f"{mcnemar_exact_p(b, c):.2f}"
            md.append(f"| {r.label} | {diff:+.0f} pt | {p_fisher:.2f} | {split} | {p_mcnemar} |")
        md += [
            "",
            "p 値の読み方: 0.05 未満なら「差がある」と言ってよい目安。"
            "それより大きいときは「差は見られなかった（50 本では区別できない）」と書く。"
            "同じチェックポイントの評価をやり直すだけでも、成否が変わるシードがある（README の「評価の再現性」）。",
        ]

    md += ["", "元データ:", ""]
    md += [f"- {r.label}: " + " + ".join(f"`{p}/results.json`" for p in r.paths) for r in results]
    (out_dir / "table.md").write_text("\n".join(md) + "\n")

    with (out_dir / "table.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# --- グラフ ----------------------------------------------------------------------------------------


def style_axes(ax: plt.Axes) -> None:
    """軸・補助線を目立たせない共通の見た目にする（データの印が主役になるように）。"""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelcolor=INK_SECONDARY, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def new_figure(n_rows: int, width: float = 7.5) -> tuple[plt.Figure, plt.Axes]:
    """条件の数に合わせた高さの図を作る（条件が増えても棒が潰れないように）。"""
    fig, ax = plt.subplots(figsize=(width, 1.2 + 0.55 * n_rows), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    return fig, ax


def plot_success_rate(results: list[RunResult], title: str, path: Path) -> None:
    """成功率の横棒と 95% 信頼区間。条件名が長いので横向きにし、上から表と同じ順に並べる。"""
    fig, ax = new_figure(len(results))
    y = np.arange(len(results))
    rates = np.array([r.success_rate for r in results]) * 100
    ci = np.array([wilson_interval(r.successes, r.n) for r in results]) * 100
    ax.barh(y, rates, height=0.5, color=SERIES, zorder=2)
    ax.errorbar(
        rates,
        y,
        xerr=[rates - ci[:, 0], ci[:, 1] - rates],
        fmt="none",
        ecolor=INK_SECONDARY,
        elinewidth=1.2,
        capsize=4,
        zorder=3,
    )
    # 値は信頼区間の右端の外に書く（棒や区間の線と重ならないように）
    for yi, rate, hi, r in zip(y, rates, ci[:, 1], results, strict=True):
        ax.text(
            hi + 1.5,
            float(yi),
            f"{rate:.0f}% ({r.successes}/{r.n})",
            va="center",
            color=INK,
            fontsize=9,
        )
    ax.set_yticks(y, [r.label for r in results])
    ax.invert_yaxis()  # 表と同じく、最初の条件（基準）を一番上にする
    ax.set_xlim(0, 115)  # 100% の棒でも右に値の文字が入る余白を取る
    ax.set_xticks(range(0, 101, 20), [f"{v}%" for v in range(0, 101, 20)])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Success rate (error bar: 95% Wilson CI)", color=INK_SECONDARY, fontsize=9)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_outcomes(results: list[RunResult], title: str, path: Path) -> None:
    """各条件の全エピソードを「どの段階で止まったか」で積み上げた横棒（合計 = エピソード数）。

    左から no_touch → … → success の順に並べるので、右端の濃い区間が成功、その左が失敗の型になる。
    どこで失敗しがちかを条件どうしで見比べるための図。
    """
    fig, ax = new_figure(len(results), width=8.5)
    y = np.arange(len(results))
    left = np.zeros(len(results))
    for i, (name, color, text_color) in enumerate(
        zip(OUTCOMES, OUTCOME_COLORS, OUTCOME_TEXT, strict=True)
    ):
        counts = np.array([r.outcome_counts()[i] for r in results])
        # 区間の境目に背景色の線を引き、隣の区間と色が接して見分けにくくならないようにする
        ax.barh(
            y,
            counts,
            left=left,
            height=0.5,
            color=color,
            edgecolor=SURFACE,
            linewidth=1.5,
            label=name,
            zorder=2,
        )
        for yi, x0, c in zip(y, left, counts, strict=True):
            if c >= 2:  # 1 本だけの区間は幅が狭く、数字が収まらないので書かない（表で見る）
                ax.text(
                    x0 + c / 2,
                    float(yi),
                    str(c),
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=8,
                )
        left += counts
    ax.set_yticks(y, [r.label for r in results])
    ax.invert_yaxis()
    ax.set_xlim(0, max(r.n for r in results))
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Episodes (stage where each episode stopped)", color=INK_SECONDARY, fontsize=9)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    # 凡例は図の下に1行で並べる（棒の上に重ならないように）
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.35 if len(results) < 3 else -0.22),
        ncol=len(OUTCOMES),
        frameon=False,
        fontsize=8,
        labelcolor=INK_SECONDARY,
    )
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_speed_vs_success(results: list[RunResult], title: str, path: Path) -> bool:
    """横軸に predict_chunk 1回の時間（中央値）、縦軸に成功率を取った散布図。

    DDPM（約 180 ms）と ACT（約 7 ms）のように桁が違うので、横軸は対数にする。
    20 ms（50Hz 制御の1ステップ）に縦線を引き、実機で間に合うかを見られるようにする。
    時間が記録されていない条件（古い results.json）は描けないので除く。描ける条件がなければ False を返す。
    """
    usable = [r for r in results if r.predict_ms is not None]
    if not usable:
        return False
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    ms = np.array([r.predict_ms for r in usable])
    rates = np.array([r.success_rate for r in usable]) * 100
    ci = np.array([wilson_interval(r.successes, r.n) for r in usable]) * 100
    ax.errorbar(
        ms,
        rates,
        yerr=[rates - ci[:, 0], ci[:, 1] - rates],
        fmt="none",
        ecolor=INK_MUTED,
        elinewidth=1,
        capsize=3,
        zorder=2,
    )
    # 点の周りに背景色の縁を付け、信頼区間の線と重なっても点の輪郭が見えるようにする
    ax.scatter(ms, rates, s=50, color=SERIES, edgecolors=SURFACE, linewidths=1.5, zorder=3)
    for x, yv, r in zip(ms, rates, usable, strict=True):
        ax.annotate(
            r.label, (x, yv), xytext=(6, 6), textcoords="offset points", fontsize=8, color=INK
        )
    ax.axvline(CONTROL_PERIOD_MS, color=INK_SECONDARY, linestyle="--", linewidth=1, zorder=1)
    # 線の説明は上端に書く（成功率の低い点は下のほうに来るので、点のラベルと重ならないように）
    ax.text(
        CONTROL_PERIOD_MS * 1.05,
        103,
        "20 ms (50 Hz control)",
        color=INK_SECONDARY,
        fontsize=8,
        va="top",
    )
    ax.set_xscale("log")
    lo, hi = min(ms.min(), CONTROL_PERIOD_MS), max(ms.max(), CONTROL_PERIOD_MS)
    ax.set_xlim(lo / 1.6, hi * 1.8)  # 右端の点のラベルが切れないよう余白を取る
    # 対数軸の既定の目盛りは「6×10⁰」のような表記になるので、1-2-5 刻みの普通の数字に置き換える
    ticks = [t for t in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000) if lo / 1.6 <= t <= hi * 1.8]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylim(0, 105)
    ax.set_xlabel(
        "predict_chunk time per call, median [ms] (log scale)", color=INK_SECONDARY, fontsize=9
    )
    ax.set_ylabel("Success rate [%]", color=INK_SECONDARY, fontsize=9)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return True


# --- 実行 ----------------------------------------------------------------------------------------


def parse_run(text: str) -> tuple[str, str]:
    """--run の「ラベル=評価の出力先」を分ける。"""
    if "=" not in text:
        raise argparse.ArgumentTypeError(f"--run は「ラベル=評価の出力先」の形で書く: {text}")
    label, path = text.split("=", 1)
    return label.strip(), path.strip()


def list_presets() -> None:
    """プリセットの一覧と、各条件の results.json が揃っているかを表示する。"""
    for name, (desc, conds) in PRESETS.items():
        print(f"{name}: {desc}")
        for label, path in conds:
            mark = "ok " if has_results(path) else "-- "
            print(f"    {mark}{label:<36} {path}")


def main() -> None:
    """コマンドライン引数で選んだ条件の組について、比較表とグラフを作る。"""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--preset", choices=sorted(PRESETS), help="決めておいた条件の組")
    parser.add_argument(
        "--run",
        type=parse_run,
        action="append",
        default=[],
        help="条件を「ラベル=評価の出力先」で直接指定する（複数回指定できる。最初が基準）",
    )
    parser.add_argument("--title", help="表とグラフの題（省略時はプリセットの説明）")
    parser.add_argument(
        "--out-dir", type=Path, help="出力先（省略時は outputs/compare/<プリセット名>/）"
    )
    parser.add_argument("--list", action="store_true", help="プリセットの一覧を表示して終わる")
    args = parser.parse_args()

    if args.list:
        list_presets()
        return
    if args.preset and args.run:
        parser.error("--preset と --run は同時に使えない")
    if args.preset:
        desc, conds = PRESETS[args.preset]
        out_dir = args.out_dir or Path("outputs/compare") / args.preset
    elif args.run:
        desc, conds = "Comparison", args.run
        if args.out_dir is None:
            parser.error("--run のときは --out-dir で出力先を指定する")
        out_dir = args.out_dir
    else:
        parser.error("--preset か --run のどちらかを指定する（一覧は --list）")
    title = args.title or desc

    # 結果がまだない条件は外して作る（夜間の評価が一部失敗しても、残りで表を作れるように）
    results = []
    for label, path in conds:
        if has_results(path):
            results.append(load_result(label, path))
        else:
            print(f"skip: {label}（{path} に results.json がないものがある）")
    if not results:
        raise SystemExit("結果が1つもないので、表とグラフを作れない")
    if results[0].label != conds[0][0]:
        print(f"注意: 基準の {conds[0][0]} の結果がないので、{results[0].label} を基準にした")

    out_dir.mkdir(parents=True, exist_ok=True)
    write_tables(results, title, out_dir)
    # 図の題は英語にする（日本語フォントがないため）。プリセット名を題に使う
    fig_title = args.preset or "comparison"
    plot_success_rate(results, f"Success rate: {fig_title}", out_dir / "success_rate.png")
    plot_outcomes(results, f"Where episodes stopped: {fig_title}", out_dir / "outcomes.png")
    has_speed = plot_speed_vs_success(
        results, f"Inference time vs. success: {fig_title}", out_dir / "speed_vs_success.png"
    )

    print((out_dir / "table.md").read_text())
    print(
        f"saved -> {out_dir}/ (table.md, table.csv, success_rate.png, outcomes.png"
        + (", speed_vs_success.png)" if has_speed else ")")
    )


if __name__ == "__main__":
    main()
