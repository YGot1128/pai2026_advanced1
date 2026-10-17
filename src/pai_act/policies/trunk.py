"""全手法で共有する部分（trunk）: 画像の特徴マップと関節角 → 観測トークン列。

    feature (B, 512, h, w) ─ 1x1 Conv で 512 → d ─ (B, d, h, w) ─ 平らにする ─ (B, h*w, d)  … 画像トークン
    state_n (B, 14)        ─ Linear で 14 → d ───────────────────────────── (B, 1, d)    … 関節角トークン
                                                         連結 → tokens (B, 1 + h*w, d)
    位置: 関節角トークンは学習するベクトル、画像トークンは 2D sinusoidal → pos (1, 1 + h*w, d)

h, w は特徴マップの縦横のマス数。ResNet18 は畳み込みとプーリングの stride を重ねて縦横を 1/32 に
縮めるので（nn/backbone.py）、特徴マップの 1 マスは入力画像のおよそ 32x32 画素の領域に対応する。
ViT のように画像を 32x32 のパッチに切り分けるわけではないが、トークン 1 個 = 画像の 1 区画、と
みなせる点は同じ（ただし畳み込みを重ねているので、1 マスの特徴は周囲の画素の情報も含む）。
入力 480x640 の画像なら h, w = 480/32, 640/32 = 15, 20 なので、画像トークンは 300 個。
baseline（policies/baseline.py）は特徴マップを空間平均して位置の情報を捨てていたが、ここでは
300 個のトークンに 2D の位置を付けたまま Transformer に渡すので、「画像のどこに何があるか」を扱える。

ヘッド（ACT / Diffusion / Flow Matching）は、この観測トークン列に自分のトークン
（ACT なら潜在変数 z のトークン）を足して Transformer に入れる。trunk を全手法で共有することで、
比較実験で違うのはヘッドだけ、と言えるようにする。

画像バックボーン（ResNet18）は trunk ではなく、ポリシー直下の `self.backbone` に置く
（policies/base.py の約束: 学習ループがパラメータ名 "backbone." で見分けて学習率を下げるため）。
そのため trunk は画像ではなく特徴マップを受け取る。関節角の正規化もポリシー側で済ませてから渡す。
"""

import torch
from torch import nn

from pai_act.nn.backbone import ResNet18Backbone
from pai_act.nn.position import sinusoidal_2d
from pai_act.policies.base import STATE_DIM


class ObservationTrunk(nn.Module):
    """特徴マップと正規化済みの関節角から、観測トークン列と位置エンコーディングを作る。

    Args:
        d_model: トークンの次元数。
        in_channels: 特徴マップのチャンネル数（ResNet18 なら 512）。
    """

    def __init__(self, d_model: int, in_channels: int = ResNet18Backbone.out_channels):
        super().__init__()
        # Step 1: 画像特徴の射影 self.image_proj: nn.Conv2d(in_channels, d_model, kernel_size=1)
        #         1x1 畳み込み = 各画素の 512 次元ベクトルに、同じ Linear(512 → d_model) をかけるのと同じ
        self.image_proj = nn.Conv2d(in_channels, d_model, kernel_size=1)

        # Step 2: 関節角の射影 self.state_proj: nn.Linear(STATE_DIM, d_model)
        self.state_proj = nn.Linear(STATE_DIM, d_model)

        # Step 3: 関節角トークンの位置ベクトル self.state_pos（学習する）:
        #         nn.Parameter で形 (1, 1, d_model)。初期値は標準正規分布の乱数（torch.randn）
        #         （nn.Embedding の重みと同じ初期化。
        #         nn.Embedding(1, d) と同じだが、テストが state_pos をテンソルとして読むので nn.Parameter にする）
        self.state_pos = nn.Parameter(torch.randn(1, 1, d_model))

        # Step 4: d_model を self.d_model として保存する（forward で sinusoidal_2d に渡すため）
        self.d_model = d_model

    def forward(
        self, feature: torch.Tensor, state_n: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            feature: (B, in_channels, h, w) バックボーンの特徴マップ。
            state_n: (B, 14) 正規化済みの関節角。

        Returns:
            tokens: (B, 1 + h*w, d_model)  [関節角トークン, 画像トークン h*w 個] の順
            pos: (1, 1 + h*w, d_model)     tokens と同じ順の位置エンコーディング
                 （バッチ内で共通なので先頭の次元は 1。足すときにブロードキャストされる）
        """
        # Step 1: 画像特徴を射影し、(B, d, h, w) → (B, h*w, d) にする
        feature_proj = self.image_proj(feature)  # (B, d, h, w)
        feature_proj = feature_proj.flatten(2).transpose(1, 2)  # (B, d, h*w)->(B, h*w, d)

        # Step 2: 関節角を射影し、(B, d) → (B, 1, d) にする
        state_proj = self.state_proj(state_n)  # (B, d)
        state_proj = state_proj.unsqueeze(1)  # (B, 1, d)

        # Step 3: [関節角, 画像] の順に、トークンの軸（dim=1）で連結して tokens にする
        tokens = torch.cat([state_proj, feature_proj], dim=1)  # ([B, 1, d] + [B, h*w, d])

        # Step 4: 画像トークンの位置 sinusoidal_2d(h, w, d_model) を作り（device は feature に合わせる）、
        #         (1, h*w, d) にしてから、関節角トークンの位置 self.state_pos と連結して pos にする
        feature_h, feature_w = feature.shape[2], feature.shape[3]
        pos = sinusoidal_2d(feature_h, feature_w, self.d_model, device=feature.device)  # (h*w, d)
        pos = pos.unsqueeze(0)  # (1, h*w, d)
        pos = torch.cat([self.state_pos, pos], dim=1)  # ([1, 1, d] + [1, h*w, d])

        return tokens, pos
