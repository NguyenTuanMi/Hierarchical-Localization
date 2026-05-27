import sys
from pathlib import Path
import torch

# ── Point to third_party/ASpanFormer ──────────────────────
# This resolves correctly regardless of where you run from
ASPANFORMER_PATH = Path(__file__).parent.parent.parent / \
                   'third_party' / 'ASpanFormer'
sys.path.insert(0, str(ASPANFORMER_PATH))

from ASpanFormer.src.ASpanFormer.aspanformer import ASpanFoormer as ASF
from ASpanFormer.src.config.default import get_cfg_defaults
from ..utils.base_model import BaseModel

class ASpanFormer(BaseModel):
    default_conf = {
        "weights": "outdoor",
        "match_threshold": 0.2,
        "max_num_matches": None,
    }
    required_inputs = ["image0", "image1"]

    def _init(self, conf):
        cfg = default_cfg
        cfg["match_coarse"]["thr"] = conf["match_threshold"]
        self.net = LoFTR_(pretrained=conf["weights"], config=cfg)

    def _forward(self, data):
        # For consistency with hloc pairs, we refine kpts in image0!
        rename = {
            "keypoints0": "keypoints1",
            "keypoints1": "keypoints0",
            "image0": "image1",
            "image1": "image0",
            "mask0": "mask1",
            "mask1": "mask0",
        }
        data_ = {rename[k]: v for k, v in data.items()}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred = self.net(data_)

        scores = pred["confidence"]

        top_k = self.conf["max_num_matches"]
        if top_k is not None and len(scores) > top_k:
            keep = torch.argsort(scores, descending=True)[:top_k]
            pred["keypoints0"], pred["keypoints1"] = (
                pred["keypoints0"][keep],
                pred["keypoints1"][keep],
            )
            scores = scores[keep]

        # Switch back indices
        pred = {(rename[k] if k in rename else k): v for k, v in pred.items()}
        pred["scores"] = scores
        del pred["confidence"]
        return pred


