"""Diagnostic exports: individual image assets have no axes, labels, or colorbars."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.transforms import Affine2D
from ..utils import write_json

plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "pdf.fonttype": 42, "svg.fonttype": "none"})


def asset(path, values, cmap="gray", vmin=0, vmax=1):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, values, cmap=cmap, vmin=vmin, vmax=vmax)


def save_figure(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def trajectory_strip(video, tracks, anchors, settings, path):
    options = settings["snapshots"]
    t, h, w = video.shape
    gain = options["trajectory_gain"]
    shown = anchors[None]+gain*(tracks-anchors[None])
    magnitude = np.linalg.norm(tracks-anchors[None], axis=-1).max(0)
    strength = np.clip(magnitude/options["motion_reference_px"], 0, 1)
    tau = np.linspace(0, 1, t)
    span, width, shear = 2.4, .7, .14
    xy = np.stack([span*tau[:, None]+width*shown[..., 0]/w, -shown[..., 1]/h+shear*shown[..., 0]/w], -1)
    fig = plt.figure(figsize=(8.4, 2.8))
    ax = fig.add_axes([.01, .01, .98, .98])
    corners = []
    for time, alpha in zip([0, t//2, t-1], [.26, .16, .1]):
        transform = Affine2D.from_values(width/w, shear/w, 0, -1/h, span*tau[time], 0)
        ax.imshow(video[time], cmap="gray", vmin=0, vmax=1, extent=(-.5, w-.5, h-.5, -.5),
                  transform=transform+ax.transData, interpolation="nearest", alpha=alpha)
        corners.extend(transform.transform([[-.5, -.5], [w-.5, h-.5]]))
    segments = np.stack([xy[:-1], xy[1:]], 2)
    colors = plt.get_cmap("viridis")((tau[:-1]+tau[1:])/2)
    order = np.argsort(strength)
    rgba = np.broadcast_to(colors, (len(order), len(colors), 4)).copy()
    rgba[:, :, 3] = (.15+.75*strength[order])[:, None]
    ax.add_collection(LineCollection(segments.transpose(1, 0, 2, 3)[order].reshape(-1, 2, 2),
        colors=rgba.reshape(-1, 4), linewidths=np.repeat(.15+1.05*strength[order], t-1)))
    bounds = np.concatenate([xy.reshape(-1, 2), np.array(corners)])
    ax.set(xlim=(bounds[:, 0].min()-.05, bounds[:, 0].max()+.05), ylim=(bounds[:, 1].min()-.05, bounds[:, 1].max()+.05), aspect="equal")
    ax.set_axis_off()
    path = Path(path)
    fig.savefig(path.with_suffix(".png"), dpi=options["dpi"], transparent=True, bbox_inches="tight", pad_inches=.01)
    fig.savefig(path.with_suffix(".svg"), transparent=True, bbox_inches="tight", pad_inches=.01)
    plt.close(fig)


def case_panels(original, prediction, mask, root, settings, roi=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    reconstructed = original*(1-mask)+prediction*mask
    error = np.abs(prediction-original)*mask
    h, w = original.shape[-2:]
    if roi is None:
        roi = [w//4, h//4, 3*w//4, 3*h//4]
        roi_kind = "central geometric ROI, NOT an anatomical heart annotation"
    else:
        roi_kind = "user specified ROI"
    x0, y0, x1, y1 = [int(v) for v in roi]
    if not 0 <= x0 < x1 <= w or not 0 <= y0 < y1 <= h:
        raise ValueError("ROI outside cached image")
    frames = settings["snapshots"]["frames"]
    fig, axes = plt.subplots(len(frames), 6, figsize=(15, 2.7*len(frames)), squeeze=False)
    for row, t in enumerate(frames):
        values = [original[t], original[t]*(1-mask[t]), np.where(mask[t], prediction[t], .5), reconstructed[t], error[t], reconstructed[t, y0:y1, x0:x1]]
        names = ["original", "masked", "masked_prediction", "reconstruction", "masked_absolute_error", "roi_reconstruction"]
        titles = ["Original", "Masked input", "Masked prediction", "Reconstruction", "Masked absolute error", "ROI"]
        for col, (value, name, title) in enumerate(zip(values, names, titles)):
            cmap = "magma" if col == 4 else "gray"
            vmax = settings["snapshots"]["error_max"] if col == 4 else 1
            asset(root/(name+"_%02d.png" % t), value, cmap, 0, vmax)
            axes[row, col].imshow(value, cmap=cmap, vmin=0, vmax=vmax)
            axes[row, col].set_title(title+"\nt=%d" % t, fontsize=9)
            axes[row, col].axis("off")
        asset(root/("prediction_%02d.png" % t), prediction[t])
        asset(root/("roi_original_%02d.png" % t), original[t, y0:y1, x0:x1])
    fig.subplots_adjust(wspace=.04, hspace=.18)
    save_figure(fig, root/"overview")
    line = (x0+x1)//2
    original_ts, reconstructed_ts = original[:, :, line].T, reconstructed[:, :, line].T
    asset(root/"time_space_original.png", original_ts)
    asset(root/"time_space_reconstruction.png", reconstructed_ts)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    for ax, image, title in zip(axes, [original_ts, reconstructed_ts], ["Original", "Reconstruction"]):
        ax.imshow(image, cmap="gray", vmin=0, vmax=1, aspect="auto", extent=[-.5, 15.5, h-.5, -.5])
        ax.set(title=title, xlabel="Phase index")
    axes[0].set_ylabel("Position along fixed line (px)")
    fig.tight_layout()
    save_figure(fig, root/"time_space_overview")
    np.savez_compressed(root/"arrays.npz", original=original, prediction=prediction, reconstruction=reconstructed, mask=mask, masked_absolute_error=error, roi=roi, line_x=line)
    write_json(root/"display.json", dict(roi=roi, roi_kind=roi_kind, line_x=line, error_range=[0, settings["snapshots"]["error_max"]],
               prediction_note="Visible-patch decoder outputs are not directly supervised; metrics only use masked pixels."))


def large_map(path, values, settings, cmap="inferno", vmax=1.):
    import cv2
    pixels = settings["snapshots"]["map_pixels"]
    asset(path, cv2.resize(values, (pixels, pixels), interpolation=cv2.INTER_NEAREST), cmap, 0, vmax)


def motion_panels(video, tracks, maps, anchors, root, settings, render_sources=None):
    root, size = Path(root), video.shape[-1]
    m_display_max = float(settings["snapshots"].get("m_display_max", 1.))
    if not np.isfinite(m_display_max) or m_display_max <= 0:
        raise ValueError("snapshots.m_display_max must be positive and finite")
    arrays = {}
    for name, path in tracks.items():
        arrays[name+"_positions"] = path
        if render_sources is None or name in render_sources:
            trajectory_strip(video, path, anchors, settings, root/(name+"_trajectories"))
    for name, (k, r, m) in maps.items():
        g = int(np.sqrt(len(m)))
        for letter, values in [("K", k), ("R", r), ("M", m)]:
            arrays[name+"_"+letter] = values
            if letter == "M" and (render_sources is None or name in render_sources):
                large_map(root/("M_"+name+".png"), values.reshape(g, g), settings,
                          vmax=m_display_max)
                if name in ["query", "dis"]:
                    large_map(root/("M.png" if name == "query" else "M_DIS_reference.png"),
                              values.reshape(g, g), settings, vmax=m_display_max)
        import cv2
        enlarged = cv2.resize(m.reshape(g, g), (size, size), interpolation=cv2.INTER_LINEAR)
        rgb = plt.get_cmap("inferno")(np.clip(enlarged/m_display_max, 0, 1))[..., :3]
        if render_sources is None or name in render_sources:
            overlay = .65*np.repeat(video[0, :, :, None], 3, -1)+.35*rgb
            pixels = settings["snapshots"]["map_pixels"]
            asset(root/("M_"+name+"_overlay.png"), cv2.resize(overlay, (pixels, pixels), interpolation=cv2.INTER_LINEAR))
    if "query" in tracks and "dis" in tracks:
        difference = np.linalg.norm(tracks["query"]-tracks["dis"], axis=-1).mean(0)
        arrays["query_minus_dis_mean_distance_px"] = difference
        large_map(root/"query_DIS_difference.png", difference.reshape(int(len(difference)**.5), -1), settings,
                  "magma", settings["snapshots"]["motion_reference_px"])
    fig, ax = plt.subplots(figsize=(5.5, 3))
    # Fixed central and peripheral spatial anchors, not automatic tissue labels.
    g = int(len(anchors)**.5)
    points = [g*(g//2)+g//2, g*(g//2)+max(0, g//2-2), 0]
    styles = ["-", "--", ":"]
    for name, path in tracks.items():
        for point, style in zip(points, styles):
            ax.plot(np.linalg.norm(path[:, point]-anchors[point], axis=-1), style, label="%s p%d" % (name, point), linewidth=1)
    ax.set(xlabel="Phase index", ylabel="Displacement from source (px)")
    ax.legend(fontsize=6, ncol=3)
    fig.tight_layout()
    save_figure(fig, root/"displacement_curves")
    np.savez_compressed(root/"motion_arrays.npz", anchors=anchors, **arrays)
    write_json(root/"motion_display.json", dict(source_phase=0, display_gain=settings["snapshots"]["trajectory_gain"], raw_arrays_unscaled=True,
         png_pixels=settings["snapshots"]["map_pixels"], map_display_interpolation="nearest; numeric grid unchanged",
         M_display_range=[0., m_display_max],
         M_definition="0.5*(clip(K/k_scale_px,0,1)^2 + clip(R/r_scale,0,1)^2); full-grid source-0 display",
         M_scales={"k_scale_px": settings["weights"]["k_scale_px"],
                   "r_scale": settings["weights"]["r_scale"]},
         trajectory_scope="Full grid only for diagnostics; training queries are masked patch centers and four intra-patch offsets",
         difference_note="Query-DIS is a difference between estimators, NOT a correction or error against ground truth",
         phase_note="First phase called source/phase 0; ED alignment is only guaranteed by the cache metadata"))


def downstream_plots(frame, metrics, classes, root, task):
    root = Path(root)
    if task == "classification":
        matrix = np.asarray(metrics["confusion"])
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(matrix, cmap="Blues")
        ax.set(xticks=range(len(classes)), yticks=range(len(classes)), xticklabels=classes, yticklabels=classes, xlabel="Predicted", ylabel="True")
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="white" if matrix[i,j] > matrix.max()/2 else "black")
        fig.tight_layout()
        save_figure(fig, root/"confusion_matrix")
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(classes, [metrics["per_class_recall"][c] or 0 for c in classes], color="#267a78")
        ax.set(ylim=(0, 1), ylabel="Recall")
        fig.tight_layout()
        save_figure(fig, root/"class_recall")
    else:
        y, p = frame.target.to_numpy(), frame.predicted.to_numpy()
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.scatter(y, p, s=16, alpha=.6, color="#267a78")
        ax.plot([0, 100], [0, 100], "--", color="gray")
        ax.set(xlabel="True EF (%)", ylabel="Predicted EF (%)", xlim=(0, 100))
        fig.tight_layout()
        save_figure(fig, root/"ef_scatter")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.scatter((y+p)/2, p-y, s=16, alpha=.6, color="#267a78")
        ax.axhline(metrics["bias"], color="black")
        for value in metrics["limits_of_agreement"] or []:
            ax.axhline(value, color="#b74943", linestyle="--")
        ax.set(xlabel="Mean true/predicted EF (%)", ylabel="Prediction - truth (percentage points)")
        fig.tight_layout()
        save_figure(fig, root/"bland_altman")
