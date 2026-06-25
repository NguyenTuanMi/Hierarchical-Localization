from pathlib import Path
from pprint import pformat

from hloc import (
    extract_features,
    match_features,
    pairs_from_retrieval,
    pairs_from_covisibility,
    match_dense
)
from hloc import triangulation, localize_sfm

dataset = Path("datasets/aachen/")  # change this if your dataset is somewhere else
images = dataset / "images_upright/"

outputs = Path("outputs/aachen/")  # where everything will be saved
sfm_pairs = outputs / "pairs-db-covis20.txt"  # top 20 most covisible in SIFT model
loc_pairs = outputs / "pairs-query-netvlad20.txt"  # top 20 retrieved by NetVLAD
reference_sfm = outputs / "sfm_gluestick"  # the SfM model we will build
results = outputs / "Aachen_hloc_gluestick_netvlad20.txt"  # the result file
num_covis = 20
num_loc = 50

# list the standard configurations available
print(f"Configs for feature extractors:\n{pformat(extract_features.confs)}")
print(f"Configs for feature matchers:\n{pformat(match_features.confs)}")

sift_model = outputs / "sfm_sift"  # existing SIFT model

# ── Retrieval ──────────────────────────────────────────────────────────
retrieval_conf = extract_features.confs["netvlad"]
retrieval_path = extract_features.main(retrieval_conf, images, outputs)

# ── SfM pairs: use covisibility from the existing SIFT model ──────────
pairs_from_covisibility.main(
    sift_model, sfm_pairs, num_matched=num_covis
)

# ── Feature extraction + matching ─────────────────────────────────────
# GlueStick handles extraction internally — no separate extract step
matcher_conf = match_dense.confs["gluestick"]
# feature_conf = None   # GlueStick is self-contained

# For GlueStick we go straight to matching
features, sfm_matches = match_dense.main(
    conf=matcher_conf,
    pairs=sfm_pairs,
    image_dir=images,        # ← image_dir IS valid in match_dense.main()
    export_dir=outputs,
    max_kps=8192,
    overwrite=False,
)

print(features)
print(sfm_matches)

# ── COLMAP triangulation ───────────────────────────────────────────────
reconstruction = triangulation.main(
    reference_sfm,
    sift_model,
    images,
    sfm_pairs,
    features,
    sfm_matches,
)

# ── Query retrieval pairs ──────────────────────────────────────────────

pairs_from_retrieval.main(
        retrieval_path,
        loc_pairs,
        num_matched=num_loc,
        query_prefix="query",
        db_model=reference_sfm,























        
)

# ── Query feature matching ─────────────────────────────────────────────
loc_features, loc_matches = match_dense.main(
    conf=matcher_conf,
    pairs=loc_pairs,
    image_dir=images,
    export_dir=outputs,
    features_ref=features,   # reuse db keypoints from SfM
    overwrite=False,
    matches=sfm_matches,
)

# ── hloc point-only localization ──────────────────────────────────────
results_point = outputs / f"Aachen_hloc_superpoint+superglue_netvlad{num_loc}.txt"
hloc_log_file = str(results_point).replace(".txt", "_logs.pkl")

localize_sfm.main(
    reconstruction=reconstruction,
    queries=dataset / "queries/*_time_queries_with_intrinsics.txt",
    retrieval=loc_pairs,
    features=loc_features,
    matches=loc_matches,
    results=results_point,
    covisibility_clustering=False,
)