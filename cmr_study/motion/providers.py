"""Absolute source->target coordinates. No global-motion residuals or identity fallback."""
import importlib.util
import json
import os
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
from ..utils import signature, file_identity, write_json
from .geometry import patch_anchors, local_points


def cache_path(row, view, kind, s, source_frames=None):
    versions = dict(opencv=cv2.__version__)
    if kind == "registration":
        import SimpleITK as sitk
        versions["sitk"] = sitk.Version_VersionString()
    keys = {"dis": ["flow_size", "fb_scale_px"],
            "registration": ["flow_size", "demons_iterations", "demons_factors", "cpu_threads"],
            "heart": [], "cotracker": ["cotracker_repo"]}[kind]
    options = {k: s["motion"][k] for k in keys}
    if kind == "cotracker":
        weights = Path(s["motion"]["cotracker_checkpoint"])
        options["checkpoint"] = file_identity(weights)
    code = Path(__file__).read_bytes().hex()
    key = signature(dict(cine=file_identity(row["cache_path"]), view=view, kind=kind,
                         options=options, model_patch=s["model"]["patch_size"], tube=s["model"]["tubelet_size"],
                         source_frames=source_frames, versions=versions, implementation=signature(code)))[:24]
    return Path(s["motion"]["cache_root"]) / kind / (key+".npz")


def sample(field, xy):
    shape = xy.shape
    pts = xy.reshape(-1, 2).astype(np.float32)
    out = cv2.remap(field, pts[:, 0].reshape(-1, 1), pts[:, 1].reshape(-1, 1),
                    interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out.reshape(*shape[:-1], *field.shape[2:])


def demons(source, target, options):
    import SimpleITK as sitk
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(options.get("cpu_threads", 1))
    fixed, moving = [sitk.GetImageFromArray(a.astype(np.float32)) for a in (source, target)]
    field = None
    for factor, iterations in zip(options["demons_factors"], options["demons_iterations"]):
        def down(image):
            if factor == 1:
                return image
            return sitk.Shrink(sitk.SmoothingRecursiveGaussian(image, factor*.5), [factor, factor])
        f, m = down(fixed), down(moving)
        if field is None:
            field = sitk.Image(f.GetSize(), sitk.sitkVectorFloat64, 2)
            field.CopyInformation(f)
        else:
            field = sitk.Resample(field, f, sitk.Transform(2, sitk.sitkIdentity), sitk.sitkLinear, 0., sitk.sitkVectorFloat64)
        reg = sitk.DiffeomorphicDemonsRegistrationFilter()
        reg.SetNumberOfIterations(iterations)
        reg.SetStandardDeviations(1.)
        reg.SetNumberOfThreads(1)
        reg.SetNumberOfWorkUnits(options.get("cpu_threads", 1))
        field = reg.Execute(f, m, field)
    return sitk.GetArrayFromImage(field).astype(np.float32)


def pair_tracks(video, kind, s, source_frames=None):
    t, h, w = video.shape
    size, patch, tube = s["motion"]["flow_size"], s["model"]["patch_size"], s["model"]["tubelet_size"]
    anchors = patch_anchors(h, patch)
    local = local_points(anchors, patch).numpy()
    points = (local+.5)*size/w-.5
    sources = np.arange(0, t, tube) if source_frames is None else np.asarray(source_frames, dtype=int)
    if len(sources) == 0 or np.any(sources < 0) or np.any(sources >= t) or len(set(sources)) != len(sources):
        raise ValueError("Invalid source phases")
    positions = np.broadcast_to(local, (len(sources), t, *local.shape)).copy()
    confidence = np.ones(positions.shape[:-1], np.float32)
    frames = [cv2.resize(frame.astype(np.float32), (size, size)) for frame in video]
    cv2.setNumThreads(1)
    for i, src in enumerate(sources):
        for dst in range(t):
            if src == dst:
                continue
            if kind == "dis":
                a, b = [(f.clip(0, 1)*255).round().astype(np.uint8) for f in (frames[src], frames[dst])]
                engine = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
                flow = engine.calc(a, b, None)
                backward = engine.calc(b, a, None)
                delta = sample(flow, points)
                fb = delta+sample(backward, points+delta)
                confidence[i, dst] = np.exp(-np.linalg.norm(fb, axis=-1)/(s["motion"]["fb_scale_px"]*size/w))
            elif kind == "registration":
                flow = demons(frames[src], frames[dst], s["motion"])
                delta = sample(flow, points)
            else:
                raise ValueError(kind)
            positions[i, dst] = (points+delta+.5)*w/size-.5
            valid = ((positions[i, dst] >= 0) & (positions[i, dst] <= w-1)).all(-1)
            confidence[i, dst] *= valid
    if not np.isfinite(positions).all():
        raise FloatingPointError("Nonfinite trajectories")
    return dict(positions=positions, confidence=confidence, anchors=anchors.numpy(), sources=sources)


def heart_map(video):
    # HeartMAE's released Farneback schedule: source step 5, phase gap 4, sigma 10.
    frames = (video.clip(0, 1)*255).round().astype(np.uint8)
    maps = []
    for a in range(0, len(frames)-4, 5):
        f = cv2.calcOpticalFlowFarneback(frames[a], frames[a+4], None, .5, 3, 15, 3, 5, 1.2, 0)
        maps.append(np.linalg.norm(f, axis=-1))
    x = cv2.GaussianBlur(np.mean(maps, axis=0), (0, 0), sigmaX=10)
    lo, hi = np.quantile(x, [.0005, .9995])
    return np.clip((x-lo)/max(hi-lo, 1e-7), 0, 1).astype(np.float32)


_TRACKERS = {}


def cotracker(video, s):
    repo = Path(s["motion"]["cotracker_repo"]).resolve()
    checkpoint = Path(s["motion"]["cotracker_checkpoint"]).resolve()
    if not (repo / "cotracker/predictor.py").is_file() or not checkpoint.is_file():
        raise FileNotFoundError("TrackMAE requires existing official CoTracker3 code/weights; set the two motion.cotracker paths. No automatic download or DIS fallback.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    key = (str(repo), str(checkpoint), device)
    if key not in _TRACKERS:
        sys.path.insert(0, str(repo))
        from cotracker.predictor import CoTrackerPredictor
        _TRACKERS[key] = CoTrackerPredictor(checkpoint=str(checkpoint)).to(device).eval()
    model = _TRACKERS[key]
    anchors = patch_anchors(video.shape[-1], s["model"]["patch_size"])
    queries = torch.cat([torch.zeros(len(anchors), 1), anchors], 1)[None].to(device)
    clip = torch.from_numpy(video.copy())[None, :, None].repeat(1, 1, 3, 1, 1).to(device)*255.
    with torch.no_grad():
        tracks, visibility = model(clip, queries=queries, backward_tracking=False)
    return dict(positions=tracks[0].cpu().numpy(), confidence=visibility[0].float().cpu().numpy(), anchors=anchors.numpy())


def prepare(row, view, kind, s, source_frames=None):
    path = cache_path(row, view, kind, s, source_frames)
    if path.is_file():
        return path
    video = np.asarray(np.load(row["cache_path"], mmap_mode="r")[view], np.float32)
    values = dict(weight=heart_map(video)) if kind == "heart" else cotracker(video, s) if kind == "cotracker" else pair_tracks(video, kind, s, source_frames)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **values)
    os.replace(tmp, path)
    write_json(path.with_suffix(".json"), dict(patient_uid=row["patient_uid"], view=view, source=row["cache_path"],
               kind=kind, coordinate_policy="absolute source-to-target pixel xy; no ECC subtraction", T=16))
    return path


def display_path(row, view, kind, s):
    """Prefer phase-0 cache; reuse full teachers if already available."""
    single = cache_path(row, view, kind, s, [0])
    full = cache_path(row, view, kind, s)
    return single if single.is_file() or not full.is_file() else full


def prepare_display(row, view, kind, s):
    path = display_path(row, view, kind, s)
    return path if path.is_file() else prepare(row, view, kind, s, [0])


def load(row, view, kind, s):
    path = cache_path(row, view, kind, s)
    if not path.is_file():
        raise FileNotFoundError("Missing motion cache. Run pipeline.prepare_motion() before training: " + str(path))
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}
