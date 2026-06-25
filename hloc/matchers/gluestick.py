import sys
from pathlib import Path

import torch

from .. import logger
from ..utils.base_model import BaseModel

gluestick_path = Path(__file__).parent / "../../third_party/GlueStick"
sys.path.append(str(gluestick_path))

from gluestick import batch_to_np
from gluestick.models.two_view_pipeline import TwoViewPipeline

gs_path = Path(__file__).parent / "../../third_party/GlueStick"


class GlueStick(BaseModel):
    default_conf = {
        "name": "two_view_pipeline",
        "model_name": "checkpoint_GlueStick_MD.tar",
        "use_lines": True,
        "max_keypoints": 1000,
        "max_lines": 300,
        "force_num_keypoints": False,
    }
    required_inputs = [
        "image0",
        "image1",
    ]

    # Initialize the line matcher
    def _init(self, conf):
        # Download the model.
        model_path = gs_path / "resources/weights/checkpoint_GlueStick_MD.tar" 
        logger.info("Loading GlueStick model...")

        gluestick_conf = {
            "name": "two_view_pipeline",
            "use_lines": True,
            "extractor": {
                "name": "wireframe",
                "sp_params": {
                    "force_num_keypoints": False,
                    "max_num_keypoints": 1000,
                },
                "wireframe_params": {
                    "merge_points": True,
                    "merge_line_endpoints": True,
                },
                "max_n_lines": 300,
            },
            "matcher": {
                "name": "gluestick",
                "weights": str(model_path),
                "trainable": False,
            },
            "ground_truth": {
                "from_pose_depth": False,
            },
        }
        gluestick_conf["extractor"]["sp_params"]["max_num_keypoints"] = conf[
            "max_keypoints"
        ]
        gluestick_conf["extractor"]["sp_params"]["force_num_keypoints"] = conf[
            "force_num_keypoints"
        ]
        gluestick_conf["extractor"]["max_n_lines"] = conf["max_lines"]
        self.net = TwoViewPipeline(gluestick_conf)

    def _forward(self, data):
        img0 = data["image0"]
        img1 = data["image1"]

        pred = self.net({"image0": img0, "image1": img1})

        # GlueStick sparse output:
        #   keypoints0: [N, 2]
        #   keypoints1: [M, 2]
        #   matches0:   [N]   → index into keypoints1, -1 = unmatched
        #   matching_scores0: [N]

        kpts0_all = pred["keypoints0"][0]       # [N, 2]
        kpts1_all = pred["keypoints1"][0]       # [M, 2]
        matches   = pred["matches0"][0]         # [N]  index or -1
        scores    = pred["match_scores0"][0] # [N]

        # Convert to dense convention: keep only matched pairs
        valid = matches > -1                    # boolean mask [N]
        kpts0 = kpts0_all[valid]               # [K, 2]
        kpts1 = kpts1_all[matches[valid]]      # [K, 2]  — index lookup
        scores_matched = scores[valid]         # [K]

        # ── match_dense.py expects exactly these three keys ──────────────
        return {
            "keypoints0": kpts0,   # ← was "kpts0", now correct
            "keypoints1": kpts1,   # ← was "kpts1", now correct
            "scores":     scores_matched,  # ← correct all along
        }