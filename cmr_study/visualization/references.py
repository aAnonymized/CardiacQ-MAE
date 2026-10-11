"""Compute model-independent phase-0 references once, before training."""
from pathlib import Path
import json
import shutil
from time import perf_counter
import numpy as np
import torch
from tqdm import tqdm
from ..motion import providers
from ..motion.geometry import components, importance, patch_anchors
from ..utils import signature, file_identity, case_seed, write_json
from .plots import motion_panels


def selected_cases(rows, s):
    return sorted(rows, key=lambda r: case_seed(r["patient_uid"], 0, s["split_seed"]))[:s["snapshots"]["patients"]]


def kinds(s):
    return [k for k in s["snapshots"]["trajectory_sources"] if k != "query"]


def reference_folder(row, view, s):
    identities = {}
    for kind in kinds(s):
        path = providers.display_path(row, view, kind, s)
        if not path.is_file():
            raise FileNotFoundError("Missing %s snapshot teacher for %s view=%d. Run pipeline.prepare_motion() BEFORE training." % (kind, row["patient_uid"], view))
        identities[kind] = file_identity(path)
    key = signature(dict(cine=file_identity(row["cache_path"]), view=view, teachers=identities,
                         snapshots=s["snapshots"], weights=s["weights"], patch=s["model"]["patch_size"],
                         plots=signature(Path(__file__).with_name("plots.py").read_text(encoding="utf-8")),
                         implementation=signature(Path(__file__).read_text(encoding="utf-8"))))[:24]
    return Path(s["motion"]["cache_root"]) / "snapshot_references" / key


def ready(folder):
    marker = folder / "ready.json"
    if not marker.is_file():
        return False
    record = json.loads(marker.read_text(encoding="utf-8"))
    return all((folder / name).is_file() for name in record["files"])


def prepare_references(rows, s):
    if not kinds(s):
        return []
    timings = []
    for row in tqdm(selected_cases(rows, s), desc="Prepare fixed snapshot references"):
        for view in s["snapshots"]["views"]:
            if view >= row["views"]:
                continue
            for kind in kinds(s):
                path = providers.display_path(row, view, kind, s)
                hit, start = path.is_file(), perf_counter()
                print("[Reference]", row["patient_uid"], "view", view, kind, "cached" if hit else "compute phase 0 only", flush=True)
                providers.prepare_display(row, view, kind, s)
                timings.append(dict(patient_uid=row["patient_uid"], view=view, stage=kind,
                                    cache_hit=hit, seconds=perf_counter()-start))
            folder = reference_folder(row, view, s)
            if ready(folder):
                continue
            start = perf_counter()
            folder.mkdir(parents=True, exist_ok=True)
            tracks, maps = {}, {}
            anchors = patch_anchors(s["data"]["input_size"], s["model"]["patch_size"])
            for kind in kinds(s):
                with np.load(providers.display_path(row, view, kind, s)) as z:
                    phase_index = int(np.flatnonzero(z["sources"] == 0)[0])
                    values = torch.from_numpy(z["positions"][phase_index].transpose(1, 0, 2, 3).copy())
                k, r = components(values, anchors, s["model"]["patch_size"])
                maps[kind] = tuple(x.numpy() for x in (k, r, importance(k, r, s["weights"])))
                tracks[kind] = values[:, :, 0].transpose(0, 1).numpy()
            video = np.array(np.load(row["cache_path"], mmap_mode="r")[view], dtype=np.float32)
            motion_panels(video, tracks, maps, anchors.numpy(), folder, s)
            write_json(folder / "ready.json", dict(patient_uid=row["patient_uid"], view=view,
                files=sorted(p.name for p in folder.iterdir() if p.is_file() and p.name != "ready.json")))
            timings.append(dict(patient_uid=row["patient_uid"], view=view, stage="reference_render",
                                cache_hit=False, seconds=perf_counter()-start))
    return timings


def require_references(rows, s):
    if not kinds(s):
        return
    for row in selected_cases(rows, s):
        for view in s["snapshots"]["views"]:
            if view < row["views"] and not ready(reference_folder(row, view, s)):
                raise FileNotFoundError("Snapshot reference images missing. Run pipeline.prepare_motion() before training.")


def copy_references(row, view, s, destination):
    if not kinds(s):
        return {}, {}
    folder = reference_folder(row, view, s)
    if not ready(folder):
        raise FileNotFoundError("Unprepared snapshot references: " + str(folder))
    record = json.loads((folder / "ready.json").read_text(encoding="utf-8"))
    # Copies, not hardlinks: the combined Query plots/arrays must not alter shared references.
    for name in record["files"]:
        shutil.copyfile(folder / name, Path(destination) / name)
    with np.load(folder / "motion_arrays.npz") as data:
        tracks = {k: data[k+"_positions"].copy() for k in kinds(s)}
        maps = {k: tuple(data[k+"_"+letter].copy() for letter in ("K", "R", "M")) for k in kinds(s)}
    return tracks, maps
