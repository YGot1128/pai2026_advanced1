"""全ポリシーの共通インタフェース。

学習ループ（exps/train.py）と評価（exps/evaluate.py）は、ここで決めた2つのメソッドしか呼ばない。
そのため、ACT / Diffusion / Flow Matching などの手法を差し替えても学習・評価のコードは1つで済み、
「手法以外の条件を揃えた比較」をコードの構造で保証できる。

    compute_loss(batch) -> {"loss": ..., その他ログしたい値}   … 学習時
    predict_chunk(obs)  -> (B, k, 14) の行動チャンク             … 推論時

入出力の単位について:
    ポリシーが外とやり取りする値（batch["state"], batch["action"], predict_chunk の戻り値）は
    すべてシミュレータと同じ単位（rad など、正規化前）。正規化はポリシーの内部で行う。
    こうしておくと、学習・評価のコードは正規化を気にしなくてよく、学習時と推論時で
    前処理が食い違う事故（train/inference skew）も起きにくい。
"""

from abc import ABC, abstractmethod

import torch
from torch import nn

from pai_act.data.normalize import Normalizer

STATE_DIM = 14  # 両腕 × (6関節 + グリッパー)
ACTION_DIM = 14


class Policy(nn.Module, ABC):
    """観測から行動チャンクを出すポリシーの基底クラス。

    サブクラスが守る約束:
        - compute_loss / predict_chunk を実装する
        - 画像バックボーンは `self.backbone` という名前の属性に置く
          （学習ループがパラメータ名 "backbone." で見分け、小さい学習率を割り当てるため）

    Args:
        chunk_size: 1回の推論で出す行動のステップ数 k。

    Attributes:
        state_norm, action_norm: 正規化の統計量。学習前に学習ループが fit() で設定する。

    Note:
        abcモジュールからabstractmethodデコレータを使うと、サブクラスが抽象メソッドを実装していない場合に
        インスタンス化時に TypeError が出るので、実装漏れを早く検出できる。
    """

    def __init__(self, chunk_size: int):
        super().__init__()
        self.chunk_size = chunk_size
        self.state_norm = Normalizer(STATE_DIM)
        self.action_norm = Normalizer(ACTION_DIM)

    @abstractmethod
    def compute_loss(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """学習バッチから損失を計算する。

        Args:
            batch: data/chunk_dataset.py の ChunkDataset が返す dict をバッチにしたもの。
                image (B, 3, H, W) uint8 / state (B, 14) / action (B, k, 14) / action_is_pad (B, k)

        Returns:
            "loss" キーに逆伝播させるスカラーを入れた dict。それ以外のキーはログ用（勾配不要）。
        """

    @abstractmethod
    def predict_chunk(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """観測から行動チャンクを予測する。

        Args:
            obs: image (B, 3, H, W) uint8 / state (B, 14)。学習バッチと同じキー・同じ形。

        Returns:
            (B, k, 14) の行動チャンク（正規化前の単位。そのままシミュレータに渡せる）。
        """


def masked_l1(pred: torch.Tensor, target: torch.Tensor, is_pad: torch.Tensor) -> torch.Tensor:
    """padding したステップを除いた L1 損失（平均絶対誤差）。

    Args:
        pred, target: (B, k, D)
        is_pad: (B, k) bool。True のステップはエピソード外の埋め草なので損失に含めない。

    ACT 原著は行動の再構成損失に L2 ではなく L1 を使う（L1 の方が細かい動きを正確に再現できたと報告）。
    """
    per_step = (pred - target).abs().mean(dim=-1)  # (B, k): 14次元の平均
    valid = (~is_pad).float()
    return (per_step * valid).sum() / valid.sum().clamp(min=1.0)
