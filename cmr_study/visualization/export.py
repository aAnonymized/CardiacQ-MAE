from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from ..data.dataset import CineDataset
from ..motion import providers
from ..motion.geometry import components, importance
from ..utils import case_seed, write_json
from .plots import case_panels, motion_panels, save_figure
from .references import selected_cases, require_references, copy_references
from .timing import timed
from ..models.masking import make_mask


@torch.no_grad()
def export_cases(model, rows, settings, seed, root, device, epoch, anatomy=False):
    device = torch.device(device)
    model.eval()
    selected = selected_cases(rows, settings)
    require_references(rows, settings)
    root = Path(root)
    roi_table = None
    if settings["snapshots"]["roi_csv"]:
        roi_table = pd.read_csv(settings["snapshots"]["roi_csv"])
    for row in tqdm(selected, desc="Export fixed validation cases"):
        def case_path(v):
            return root / (row["patient_uid"].replace(":", "_")+"_view%d" % v)
        pending = [v for v in settings["snapshots"]["views"] if v < row["views"] and not (case_path(v)/"complete.json").is_file()]
        if not pending:
            continue
        with timed(root, row["patient_uid"], pending, "load_cine"):
            video_all = np.asarray(np.load(row["cache_path"], mmap_mode="r"), np.float32)
        with timed(root, row["patient_uid"], pending, "reconstruction_inference", device):
            videos = torch.from_numpy(np.stack([video_all[v] for v in pending]))[:, None].to(device)
            mask_tokens = torch.cat([make_mask(1, 16//settings["model"]["tubelet_size"], len(model.anchors),
                settings["model"]["mask_ratio"], device, seed=case_seed(row["patient_uid"], v, seed)) for v in pending])
            out = model(dict(video=videos), epoch=epoch-1, eval_seed=0, force_mask=mask_tokens, auxiliary=False)
            predictions = model.backbone.unpatchify(out["prediction"])[:, 0].float().cpu().numpy()
            masks = model.backbone.unpatchify(out["mask"][:, :, None].expand_as(out["prediction"]))[:, 0].cpu().numpy()
        contexts = None
        if model.query is not None and "query" in settings["snapshots"]["trajectory_sources"]:
            contexts = out.get("context")
            if contexts is None:
                with timed(root, row["patient_uid"], pending, "query_full_context", device):
                    contexts = model.backbone.features(videos)
        labels = None
        label_error = None
        if anatomy:
            from ..evaluation.anatomy import aligned_labels, contour_propagation
            try:
                labels = aligned_labels(row, video_all.shape[-1])
            except (FileNotFoundError, KeyError, OSError) as exc:
                label_error = str(exc)
        for n, view in enumerate(pending):
            case = case_path(view)
            case.mkdir(parents=True, exist_ok=True)
            video = video_all[view]
            prediction, mask = predictions[n], masks[n]
            roi = None
            if roi_table is not None:
                found = roi_table[(roi_table.patient_uid == row["patient_uid"]) & (roi_table.view == view)]
                if len(found) != 1:
                    raise ValueError("ROI CSV must have exactly one entry for each exported patient/view")
                roi = found.iloc[0][["x0", "y0", "x1", "y1"]].tolist()
            with timed(root, row["patient_uid"], view, "reconstruction_figures_and_arrays"):
                case_panels(video, prediction, mask, case, settings, roi)
            with timed(root, row["patient_uid"], view, "copy_precomputed_motion_references"):
                tracks, maps = copy_references(row, view, settings, case)
            anchors = model.anchors.cpu().numpy()
            for name in settings["snapshots"]["trajectory_sources"]:
                if name == "query" and contexts is not None:
                    ids = torch.arange(len(anchors), device=device)
                    with timed(root, row["patient_uid"], view, "query_trajectory_inference", device):
                        values = model.predict_tracks(contexts[n:n+1], ids, torch.zeros_like(ids)).float()
                else:
                    continue
                k, r = components(values, model.anchors, settings["model"]["patch_size"])
                m = importance(k, r, settings["weights"])
                maps[name] = tuple(x.cpu().numpy() for x in (k, r, m))
                tracks[name] = values[:, :, 0].transpose(0, 1).cpu().numpy()
            if "query" in tracks:
                with timed(root, row["patient_uid"], view, "query_and_combined_motion_figures"):
                    motion_panels(video, tracks, maps, anchors, case, settings, render_sources={"query"})
                if "dis" in tracks:
                    query_vector = tracks["query"] - anchors[None]
                    dis_vector = tracks["dis"] - anchors[None]
                    query_k = np.linalg.norm(query_vector, axis=-1).mean(0)
                    dis_k = np.linalg.norm(dis_vector, axis=-1).mean(0)
                    epe = np.linalg.norm(tracks["query"] - tracks["dis"], axis=-1).mean(0)
                    correlation = (float(np.corrcoef(query_k, dis_k)[0, 1])
                                   if query_k.std() > 1e-8 and dis_k.std() > 1e-8 else None)
                    top10 = dis_k >= np.quantile(dis_k, .90)
                    grid = int(np.sqrt(len(anchors)))
                    border = np.zeros((grid, grid), dtype=bool)
                    border[[0, -1], :] = True
                    border[:, [0, -1]] = True
                    query_r = maps["query"][1].reshape(grid, grid)
                    write_json(case/"query_quality.json", dict(
                        k_correlation=correlation,
                        all_epe_px=float(epe.mean()),
                        dis_top10_epe_px=float(epe[top10].mean()),
                        query_k_mean_px=float(query_k.mean()),
                        dis_k_mean_px=float(dis_k.mean()),
                        query_top10_k_mean_px=float(query_k[top10].mean()),
                        dis_top10_k_mean_px=float(dis_k[top10].mean()),
                        query_r_border_mean=float(query_r[border].mean()),
                        query_r_interior_mean=float(query_r[~border].mean()),
                        interpretation=("Diagnostic full-grid Query-vs-DIS agreement; "
                                        "DIS is pseudo-supervision, not ground truth."),
                    ))
            if labels is not None:
                contour_propagation(video, labels, view, tracks, maps, anchors, case, settings["model"]["patch_size"])
                regions = labels["ed"][view]
                rec_rows = []
                for label in range(4):
                    select = (regions == label) & mask[0].astype(bool)
                    rec_rows.append(dict(label=label, masked_pixels=int(select.sum()),
                        masked_mse=float(np.mean((prediction[0][select]-video[0][select])**2)) if select.any() else None))
                pd.DataFrame(rec_rows).to_csv(case/"ED_region_reconstruction_error.csv", index=False)
            write_json(case/"complete.json", dict(patient_uid=row["patient_uid"], view=view, epoch=epoch, method=model.method,
                query_context="visible_tokens" if "context" in out else "full_video",
                architecture=getattr(model, "architecture_version", "legacy_maest"),
                query_display_scope="Full spatial grid at source phase 0; diagnostic, not the sparse training selection" if contexts is not None else None,
                tgwr_training_scope=("Selected masked tubelets; K uses all 16 center phases; "
                                     "R/Jacobian use four uniformly spaced relative phases; "
                                     "snapshot display may query the complete five-point grid"
                                     if getattr(model, "architecture_version", "") ==
                                     "shared_fullcycle_center_sampled_jacobian_three_loss_v4" else None),
                objective="L_rec + lambda_DIS*(L_coordinate + lambda_J*L_Jacobian) + lambda_DPCC*L_DPCC" if getattr(model, "architecture_version", "") in {"shared_fullcycle_three_loss_jacobian_v3", "shared_fullcycle_center_sampled_jacobian_three_loss_v4"} else None,
                metrics=dict(masked_mse=float(np.mean((prediction[mask.astype(bool)]-video[mask.astype(bool)])**2)),
                             mean_fill_mse=float(np.mean((video[mask.astype(bool)]-video[~mask.astype(bool)].mean())**2))),
                anatomy_status=label_error or ("evaluated" if labels else "No anatomy labels; no tissue-specific claims")))
    write_json(root/"complete.json", dict(epoch=epoch, patients=[r["patient_uid"] for r in selected], requested=settings["snapshots"]["patients"]))
    timing_path = root/"export_timings.jsonl"
    if timing_path.is_file():
        timing = pd.read_json(timing_path, lines=True)
        timing.to_csv(root/"export_timings.csv", index=False)
        print(timing.groupby("stage").seconds.agg(["count", "sum", "mean"]).to_string(), flush=True)


def compare_exports(directories, output, settings):
    """Only compare identical cases and byte-identical originals/masks."""
    from matplotlib import pyplot as plt
    directories = {name: Path(path) for name, path in directories.items()}
    sets = [set(p.parent.name for p in directory.glob("*/arrays.npz")) for directory in directories.values()]
    if not sets or any(not x for x in sets):
        raise FileNotFoundError("Missing snapshot arrays for a requested method")
    if any(x != sets[0] for x in sets[1:]):
        raise ValueError("Snapshot case sets differ; finish exports for every method first")
    for case in sorted(sets[0]):
        data = {name: dict(np.load(path/case/"arrays.npz")) for name, path in directories.items()}
        reference = next(iter(data.values()))
        if any(not np.array_equal(d["mask"], reference["mask"]) or not np.array_equal(d["original"], reference["original"]) for d in data.values()):
            raise ValueError("Different image/mask: " + case)
        rows = []
        for name, d in data.items():
            select = d["mask"].astype(bool)
            rows.append(dict(method=name, masked_mse=float(np.mean((d["prediction"][select]-d["original"][select])**2))))
        out = Path(output)/case
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out/"masked_metrics.csv", index=False)
        for t in settings["snapshots"]["frames"]:
            fig, axes = plt.subplots(2, len(data)+2, figsize=(3*(len(data)+2), 6), squeeze=False)
            for ax in axes.flat:
                ax.axis("off")
            axes[0,0].imshow(reference["original"][t], cmap="gray", vmin=0, vmax=1)
            axes[0,0].set_title("Original")
            axes[0,1].imshow(reference["original"][t]*(1-reference["mask"][t]), cmap="gray", vmin=0, vmax=1)
            axes[0,1].set_title("Shared mask")
            for col, (name, d) in enumerate(data.items(), 2):
                axes[0,col].imshow(d["reconstruction"][t], cmap="gray", vmin=0, vmax=1)
                axes[0,col].set_title(name)
                axes[1,col].imshow(d["masked_absolute_error"][t], cmap="magma", vmin=0, vmax=settings["snapshots"]["error_max"])
            fig.subplots_adjust(wspace=.04, hspace=.02)
            save_figure(fig, out/("comparison_t%02d" % t))
        line = int(reference["line_x"])
        fig, axes = plt.subplots(1, len(data)+1, figsize=(3*(len(data)+1), 3.5), squeeze=False, sharey=True)
        images = [("Original",reference["original"])] + [(name,d["reconstruction"]) for name,d in data.items()]
        for ax, (name, array) in zip(axes[0], images):
            ax.imshow(array[:, :, line].T,cmap="gray",vmin=0,vmax=1,aspect="auto")
            ax.set(title=name,xlabel="Phase index")
        axes[0,0].set_ylabel("Fixed-line position (px)")
        fig.tight_layout()
        save_figure(fig,out/"time_space_comparison")
