"""ED/ES labels are read ONLY for evaluation. They never enter pretraining."""
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from matplotlib import pyplot as plt
from ..utils import write_json
from ..visualization.plots import save_figure


def aligned_labels(row, size):
    import nibabel as nib
    from nibabel.processing import resample_from_to
    directory = Path(row["cache_path"]).parent
    meta = json.loads((directory/"metadata.json").read_text())
    references = meta.get("references", {})
    if "ed" not in references or "es" not in references:
        raise FileNotFoundError("V5 ED/ES evaluation labels/geometry unavailable")
    path = Path(row["cine_path"])
    if row.get("cine_format") == "frame_series":
        paths = sorted(p for p in path.glob(row["patient_id"]+"_frame*.nii.gz") if not p.name.endswith("_gt.nii.gz"))
        path = paths[0]
    image = nib.as_closest_canonical(nib.load(path))
    spacing = nib.affines.voxel_sizes(image.affine)
    target_spacing = float(np.linalg.norm(np.array(references["ed"]["processed_affine"])[:3, 0]))
    direction = image.affine[:3, :3]/spacing[None]
    resampled = np.maximum(1, np.rint(np.array(image.shape[:2])*spacing[:2]/target_spacing).astype(int))
    affine = image.affine.copy()
    affine[:3, :2] = direction[:, :2]*target_spacing
    affine[:3, 3] = (affine @ np.array([(resampled[0]-size)//2, (resampled[1]-size)//2, 0, 1]))[:3]
    shape = (size, size, image.shape[2])
    result = {}
    for phase in ["ed", "es"]:
        label_path = references[phase]["label_path"]
        volume = np.rint(np.asarray(resample_from_to(nib.load(label_path), (shape, affine), order=0).dataobj)).astype(np.uint8)
        if row["dataset"] != "acdc":
            original = volume.copy()
            volume[original == 1] = 3
            volume[original == 3] = 1
        result[phase] = volume.transpose(2, 0, 1)[meta["slice_indices"]]
    result["es_position"] = float(meta["es_position"])
    return result


def contour_propagation(video, labels, view, tracks, maps, anchors, root, patch):
    root = Path(root)
    ed, es = labels["ed"][view], labels["es"][view]
    pos = labels["es_position"]
    t0, frac = int(np.floor(pos)) % 16, pos % 1
    t1 = (t0+1) % 16
    target_image = video[t0]*(1-frac)+video[t1]*frac
    g = int(len(anchors)**.5)
    records = []
    colors = {1: "#43a68b", 2: "#e2b942", 3: "#ce5274"}
    for name, trajectory in tracks.items():
        target = trajectory[t0]*(1-frac)+trajectory[t1]*frac
        flow = (target-anchors).reshape(g, g, 2)
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(target_image, cmap="gray", vmin=0, vmax=1)
        for label in [1, 2, 3]:
            source_mask = (ed == label).astype(np.uint8)
            contours, _ = cv2.findContours(source_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            truth = (es == label).astype(np.uint8)
            if not contours or not truth.any():
                continue
            boundary = truth-cv2.erode(truth, np.ones((3, 3), np.uint8))
            distances = distance_transform_edt(boundary == 0).astype(np.float32)
            pts = np.concatenate([c[:, 0] for c in contours]).astype(np.float32)[::3]
            gx, gy = (pts[:, 0]+.5)/patch-.5, (pts[:, 1]+.5)/patch-.5
            disp = cv2.remap(flow, gx[:, None], gy[:, None], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).reshape(-1, 2)
            mapped = pts+disp
            valid = ((mapped >= 0) & (mapped <= video.shape[-1]-1)).all(-1)
            errors = cv2.remap(distances, mapped[:, 0:1], mapped[:, 1:2], cv2.INTER_LINEAR).ravel()
            ax.contour(truth, levels=[.5], colors=[colors[label]], linewidths=.8)
            ax.scatter(mapped[:, 0], mapped[:, 1], s=3, color=colors[label])
            records.append(dict(source=name, label=label, points=len(pts), in_bounds_fraction=float(valid.mean()),
                                mean_boundary_distance_px=float(errors[valid].mean()) if valid.any() else None))
        ax.axis("off")
        fig.savefig(root/(name+"_ED_ES_points.png"), dpi=180, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
    pd.DataFrame(records).to_csv(root/"ED_ES_boundary_metrics.csv", index=False)
    region_rows = []
    centers = np.rint(anchors).astype(int)
    regions = ed[centers[:, 1], centers[:, 0]]
    for name, (k, r, m) in maps.items():
        for label in range(4):
            select = regions == label
            if select.any():
                region_rows.append(dict(source=name, label=label, patches=int(select.sum()), K=float(k[select].mean()), R=float(r[select].mean()), M=float(m[select].mean())))
    pd.DataFrame(region_rows).to_csv(root/"M_anatomical_regions.csv", index=False)
    distribution, names, raw_rows = [], [], []
    for name, (_, _, m) in maps.items():
        for label in range(4):
            selected = m[regions == label]
            if len(selected):
                distribution.append(selected)
                names.append(name+"/"+["BG", "RV", "MYO", "LV"][label])
                raw_rows.extend(dict(source=name, label=label, M=float(value)) for value in selected)
    pd.DataFrame(raw_rows).to_csv(root/"M_region_distribution.csv", index=False)
    if distribution:
        fig, ax = plt.subplots(figsize=(max(6, .65*len(names)), 3.8))
        ax.boxplot(distribution, showfliers=False)
        ax.set_xticks(range(1, len(names)+1), names, rotation=45, ha="right")
        ax.set(ylabel="Motion importance M", ylim=(0, 1))
        fig.tight_layout()
        save_figure(fig, root/"M_region_distribution")
    write_json(root/"anatomy_evaluation.json", dict(es_position=pos, labels="0 background, 1 RV, 2 myocardium, 3 LV",
        policy="Same ED contours; same bilinear grid displacement and temporal interpolation for every method; no display gain",
        limitations="Boundary distance is not ground-truth point tracking accuracy. Cache phase interpolation only approximates the acquired ES image."))
