"""比較実験の対象となるポリシー。

画像・状態から観測トークン列を作る部分 (trunk) を全手法で共有し、
行動を生成するヘッド（CVAE / Diffusion / Flow Matching）だけを差し替える。

    base.py      全ポリシーの共通インタフェース（compute_loss / predict_chunk）
    baseline.py  パイプライン検証用の仮モデル（ResNet18 + MLP）
    trunk.py     全手法で共有する部分: 特徴マップ・関節角 → 観測トークン列
    act.py       ACT（CVAE ヘッド）
    denoiser.py  Diffusion / Flow Matching で共有するデノイザ
    diffusion.py Diffusion ヘッド（DDPM で学習、DDPM / DDIM でサンプリング）
    flow_matching.py  Flow Matching ヘッド（速度を学習、Euler / Heun で ODE を解く）
"""
