import sys
from pathlib import Path
import torch

sys.path.append(str(Path(__file__).parent / "../../third_party"))

from ASpanFormer.src.ASpanFormer.aspanformer import ASpanFormer as ASpanFormer_
from ASpanFormer.src.config.default import get_cfg_defaults
from ..utils.base_model import BaseModel
from .. import logger

ASPANF0RMER_PATH = Path(__file__).parent / "../../third_party/ASpanFormer"

class ASpanFormer(BaseModel):
    default_conf = {
        "model_name": "outdoor.ckpt",
        "match_threshold": 0.2,
        "sinkhorn_iterations": 20,
        "max_num_matches": 2048,
        "config_path": ASPANF0RMER_PATH / "configs/aspan/outdoor/aspan_test.py",
    }
    required_inputs = ["image0", "image1"]

    def _init(self, conf):
        config = get_cfg_defaults()
        config.merge_from_file(conf["config_path"])
        _config = lower_config(config)

        _config["aspan"]["match_coarse"]["thr"] = conf["match_threshold"]
        _config["aspan"]["match_coarse"]["skh_iters"] = conf["sinkhorn_iterations"]

        self.net = ASpanFormer_(config=_config['aspan'])

        weights_path = ASPANF0RMER_PATH / "weights/outdoor.ckpt"

        state_dict = torch.load(weights_path, map_location='cpu')['state_dict']
        self.net.load_state_dict(state_dict, strict=False)
        logger.info("Loaded Aspanformer model")
        

    def _forward(self, data):
        data_ = {
            "image0" : data["image0"],
            "image1" : data["image1"]
        }
        
        # Activate image resize 
        self.net(data_, online_resize=True)

        pred = {
            "keypoints0": data_["mkpts0_f"],
            "keypoints1": data_["mkpts1_f"],
            "mconf": data_["mconf"],
        }
        scores = pred["mconf"]

        top_k = self.conf["max_num_matches"]
        if top_k is not None and len(scores) > top_k:
            keep = torch.argsort(scores, descending=True)[:top_k]
            pred["keypoints0"], pred["keypoints1"] = (
                pred["keypoints0"][keep],
                pred["keypoints1"][keep],
            )
            scores = scores[keep]
        return pred


