"""Paper pretraining recipe; numerical defaults match the supplied notebook."""
from copy import deepcopy

METHODS = ("videomae", "heartmae", "trackmae", "cineqmae")
PUBLIC_DATASETS = ("acdc", "mnm1", "mnm2")


def default_settings():
    return deepcopy({
        "pretrain_source": "private",
        "private_root": "/public/home/chenweilin/CMRDataSet",
        "private_train_csv": "train_clean.csv", "private_test_csv": "test_clean.csv",
        "private_cache": "cmr_csv_cache_native_xy_v2",
        "private_previous_manifest": None,
        "native_preprocess": {"policy": "native_xy_fit_pad_v2", "slice_offsets": [-2, 0, 2],
                              "max_slices": 128, "max_frames": 200, "percentiles": [1., 99.]},
        "skip_missing_cache": True,
        "split_seed": 0, "val_fraction": .2, "label_map": {"AMI": "MI", "CMI": "MI"},
        "public_raw_root": "/data/SAX",
        "public_raw_roots": {"acdc": None, "mnm1": None, "mnm2": None},
        "public_cache_root": "./public_cmr_cache",
        "public_preprocess": {"target_spacing_xy": 1.5, "slice_offsets": [-2, 0, 2]},
        "output_root": "./training_outputs", "run_name": "cineq_paper_private", "resume": False,
        "methods": ["cineqmae"], "seeds": [0],
        "device": "auto", "amp": False, "allow_tf32": False, "num_workers": 4,
        "evaluation": {"batch_size": 32, "num_workers": 0},
        "data": {"num_frames": 16, "input_size": 224, "intensity_jitter": .1, "horizontal_flip": .5},
        "model": {"embed_dim": 768, "depth": 12, "num_heads": 12, "decoder_dim": 512,
                  "decoder_depth": 8, "decoder_heads": 16, "patch_size": 16, "tubelet_size": 4,
                  "mask_ratio": .75, "mask_strategy": "random", "norm_pix_loss": False},
        # dim/heads/depth are used only by the legacy non-shared query implementation.
        # cineQMAE queries use shared_decoder.dim/depth/heads below.
        "query": {"dim": 128, "heads": 4, "depth": 2, "chunk_size": 2048, "loss_weight": 1.,
                  "dis_loss_weight": 1., "jacobian_loss_weight": .1, "jacobian_beta": .1,
                  "max_displacement_px": 48., "phase_mode": "fixed_two_thirds", "confidence_min": .1,
                  "query_fraction": .10, "top_share": .70},
        "shared_decoder": {"dim": 512, "depth": 8, "heads": 16, "dpcc_loss_weight": .2},
        "sparse_paths": {"phase_split_fraction": 2./3., "full_cycle_weight": .5,
                         "composed_endpoint_weight": .5, "jacobian_phase_offsets": [2, 6, 10, 14]},
        "weights": {"alpha": 1., "ramp_epochs": 50, "add_baseline": False,
                    "k_scale_px": 1.2, "r_scale": .07},
        "motion": {"cache_root": "./motion_cache", "flow_size": 112, "fb_scale_px": 2., "cpu_threads": 1,
                   "demons_iterations": [40, 30, 20], "demons_factors": [4, 2, 1],
                   "cotracker_repo": "./vendor/cotracker", "cotracker_checkpoint": "./weights/scaled_offline.pth"},
        "trackmae": {"loss_weight": 1., "motion_visible_fraction": .5, "normalize_motion": "patch"},
        "pretrain": {"validate": False, "epochs": 400, "batch_size": 32, "accum_steps": 1, "lr": 1e-4,
                     "min_lr": 1e-6, "warmup_epochs": 10, "weight_decay": .05, "grad_clip": 1.,
                     "max_consecutive_amp_skips": 8},
        "snapshots": {"enabled": True, "every_epochs": 50, "patients": 25, "views": [0, 1, 2],
                      "frames": [0, 8, 15], "trajectory_gain": 2.5, "motion_reference_px": 12.,
                      "error_max": .3, "dpi": 180, "roi_csv": None,
                      "trajectory_sources": ["dis", "registration", "query"], "map_pixels": 896,
                      "m_display_max": 1.},
    })


def validate(settings):
    if settings["pretrain_source"] not in ("private", "public"):
        raise ValueError("pretrain_source must be 'private' or 'public'")
    if not settings["methods"] or set(settings["methods"]) - set(METHODS):
        raise ValueError("Choose methods from " + str(METHODS))
    if not settings["seeds"] or len(set(settings["seeds"])) != len(settings["seeds"]):
        raise ValueError("seeds must be a nonempty list of distinct integers")
    data, model, train = settings["data"], settings["model"], settings["pretrain"]
    if (data["num_frames"] != 16 or data["input_size"] % model["patch_size"] or
            data["num_frames"] % model["tubelet_size"]):
        raise ValueError("Use T=16 and sizes divisible by patch/tubelet size")
    if not 0 < model["mask_ratio"] < 1 or model["norm_pix_loss"]:
        raise ValueError("Use raw-intensity reconstruction and 0 < mask_ratio < 1")
    if min(train["epochs"], train["batch_size"], train["accum_steps"]) < 1:
        raise ValueError("epochs, batch_size and accum_steps must be positive")
    if not 0 < settings["val_fraction"] < 1:
        raise ValueError("val_fraction must lie between 0 and 1")
    if settings["snapshots"]["every_epochs"] < 1:
        raise ValueError("snapshots.every_epochs must be positive")
