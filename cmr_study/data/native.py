"""Private SAX: preserve native X-Y planes and the original Z slice axis."""
from pathlib import Path
import json
import errno
from time import perf_counter
import cv2
import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm
from ..utils import signature, file_identity, write_json

POLICY = "native_xy_fit_pad_v2"


def resolve(root, path):
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (Path(root) / path).resolve()


def csv_sources(s):
    rows, ids, paths = [], set(), set()
    for split, key in (("train", "private_train_csv"), ("test", "private_test_csv")):
        table = pd.read_csv(resolve(s["private_root"], s[key]), dtype=str).fillna("")
        if not {"sax_path", "disease"}.issubset(table.columns):
            raise ValueError("CSV requires sax_path,disease")
        for item in table.to_dict("records"):
            path = resolve(s["private_root"], item["sax_path"])
            pid = item.get("patient_id") or path.parent.name
            if pid in ids or str(path) in paths:
                raise ValueError("Duplicate private patient/path across CSVs: " + pid)
            ids.add(pid)
            paths.add(str(path))
            rows.append(dict(item, patient_id=pid, sax_path=str(path), csv_split=split))
    return rows


def native_key(row, s):
    contract = dict(policy=POLICY, source=file_identity(row["sax_path"]),
                    options=s["native_preprocess"], size=s["data"]["input_size"],
                    frames=s["data"]["num_frames"], ed_index=row.get("ed_index", ""),
                    implementation=signature(Path(__file__).read_text(encoding="utf-8")))
    return signature(contract)[:24], contract


def load_native(row, s):
    image = nib.load(row["sax_path"])
    shape, options = image.shape, s["native_preprocess"]
    if (len(shape) != 4 or min(shape) < 1 or shape[3] < 2 or
            shape[2] > options["max_slices"] or shape[3] > options["max_frames"]):
        raise ValueError("Expected native X,Y,Z,T; refusing guessed axis swaps: " + str(shape))
    spacing = np.asarray(image.header.get_zooms()[:3], dtype=float)
    if (not np.isfinite(image.affine).all() or not np.isfinite(spacing).all() or
            np.any(spacing <= 0) or not np.allclose(spacing, nib.affines.voxel_sizes(image.affine), rtol=.01, atol=.01)):
        raise ValueError("Invalid/inconsistent source geometry; no automatic affine correction")
    # Read gzip once rather than decompressing it separately for each slice/frame.
    values = image.get_fdata(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Source contains NaN/Inf")
    indices = np.clip(shape[2] // 2 + np.asarray(options["slice_offsets"], int), 0, shape[2] - 1)
    native = np.stack([values[:, :, z, :].transpose(2, 1, 0) for z in indices])
    return image, native, indices, spacing


def convert_native(row, s):
    image, native, indices, spacing = load_native(row, s)
    _, nt, height, width = native.shape
    lo, hi = np.percentile(native, s["native_preprocess"]["percentiles"])
    if not np.isfinite([lo, hi]).all() or hi - lo <= 1e-8:
        raise ValueError("Constant selected cine; inspect original intensities")
    normalized = np.clip((native - lo) / (hi - lo), 0, 1)
    size = s["data"]["input_size"]
    factor = size / max(width * spacing[0], height * spacing[1])
    out_w = max(1, min(size, round(width * spacing[0] * factor)))
    out_h = max(1, min(size, round(height * spacing[1] * factor)))
    left, top = (size - out_w) // 2, (size - out_h) // 2
    resized = np.zeros((len(indices), nt, size, size), np.float32)
    interpolation = cv2.INTER_AREA if out_w <= width and out_h <= height else cv2.INTER_LINEAR
    for v in range(len(indices)):
        for t in range(nt):
            resized[v, t, top:top+out_h, left:left+out_w] = cv2.resize(normalized[v, t], (out_w, out_h), interpolation=interpolation)
    ed = str(row.get("ed_index", "")).strip()
    if ed and (not ed.isdigit() or not 0 <= int(ed) < nt):
        raise ValueError("ed_index must be an explicitly supplied zero-based source frame")
    start = int(ed) if ed else 0
    times = (start + np.arange(s["data"]["num_frames"]) * nt / s["data"]["num_frames"]) % nt
    lower, upper = np.floor(times).astype(int), (np.floor(times).astype(int) + 1) % nt
    alpha = (times - lower).astype(np.float32)[None, :, None, None]
    cine = resized[:, lower] * (1 - alpha) + resized[:, upper] * alpha
    # Columns describe output X,Y,Z; storage remains S,T,H,W, with H=Y and W=X.
    matrix = np.eye(4)
    matrix[0, 0], matrix[1, 1] = width / out_w, height / out_h
    matrix[0, 3], matrix[1, 3] = (.5-left)*width/out_w-.5, (.5-top)*height/out_h-.5
    metadata = dict(policy=POLICY, patient_id=row["patient_id"], source=row["sax_path"],
        source_identity=file_identity(row["sax_path"]), source_shape=list(image.shape),
        source_spacing=spacing.tolist(), source_affine=image.affine.tolist(),
        processed_affine=(image.affine @ matrix).tolist(), slice_indices=indices.tolist(),
        slice_axis="Original NIfTI axis 2; never canonicalized", layout="STHW",
        spatial_policy="Fit native XY field of view into square; preserve physical aspect; pad only; never reslice Z",
        content_box_xyxy=[left, top, left+out_w, top+out_h],
        temporal_source_indices=times.tolist(), temporal_policy="Uniform periodic interpolation over full source cycle",
        temporal_start_index=start, ed_aligned=bool(ed), normalization_window=[float(lo), float(hi)],
        affine_note="Matching spacing does not certify anatomy; preview source and cache before training")
    return cine.astype(np.float32), metadata


def prepare_case(row, s):
    key, contract = native_key(row, s)
    folder = resolve(s["private_root"], s["private_cache"]) / "cine" / key
    path, metadata_path = folder / "cine.npy", folder / "metadata.json"
    if path.is_file() and metadata_path.is_file():
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        old = np.load(path, mmap_mode="r", allow_pickle=False)
        expected = (len(s["native_preprocess"]["slice_offsets"]), 16, s["data"]["input_size"], s["data"]["input_size"])
        if meta.get("contract") == contract and tuple(old.shape) == expected and meta.get("cache_identity") == file_identity(path):
            return path, "reused"
        raise ValueError("Existing native cache failed identity/shape verification: " + str(folder))
    cine, metadata = convert_native(row, s)
    if not np.isfinite(cine).all() or cine.min() < 0 or cine.max() > 1:
        raise ValueError("Invalid generated cine")
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / "cine.tmp.npy"
    np.save(tmp, cine, allow_pickle=False)
    tmp.replace(path)
    write_json(metadata_path, dict(metadata, contract=contract, cache_identity=file_identity(path)))
    return path, "created"


def prepare_private(s):
    root = resolve(s["private_root"], s["private_cache"])
    if root == resolve(s["private_root"], "cmr_csv_cache"):
        raise ValueError("Use a NEW private_cache directory, not old cmr_csv_cache")
    root.mkdir(parents=True, exist_ok=True)
    report = []
    for row in tqdm(csv_sources(s), desc="Native SAX train+test caches"):
        start = perf_counter()
        try:
            path, status = prepare_case(row, s)
            event = dict(row, cache_path=str(path), status=status, reason="")
        except (ValueError, OSError, nib.filebasedimages.ImageFileError) as exc:
            if isinstance(exc, OSError) and exc.errno in (errno.ENOSPC, errno.EACCES, errno.ENOMEM, errno.EROFS):
                raise
            event = dict(row, cache_path="", status="failed", reason=str(exc))
            print("NATIVE CACHE FAILED:", row["patient_id"], str(exc), flush=True)
            if not s["skip_missing_cache"]:
                raise
        event["seconds"] = perf_counter() - start
        report.append(event)
        pd.DataFrame(report).to_csv(root / "native_prepare_report.csv", index=False)
    frame = pd.DataFrame(report)
    frame[frame.status != "failed"].to_csv(root / "native_cache_index.csv", index=False)
    print(frame.groupby(["csv_split", "status"]).size().to_string())
    print("Native cache:", root, "| original CSVs/NIfTIs/old cache unchanged")
    return frame


def preview_private(s, patient_id=None):
    """One notebook call: native source and corrected cache side by side."""
    from matplotlib import pyplot as plt
    from IPython.display import display
    row = next((r for r in csv_sources(s) if patient_id is None or r["patient_id"] == patient_id), None)
    if row is None:
        raise ValueError("Patient absent from configured CSVs: " + str(patient_id))
    path, status = prepare_case(row, s)
    image, native, indices, _ = load_native(row, s)
    cache = np.load(path, mmap_mode="r")
    meta = json.loads(path.with_name("metadata.json").read_text(encoding="utf-8"))
    lo, hi = meta["normalization_window"]
    native = np.clip((native-lo)/(hi-lo), 0, 1)
    fig, axes = plt.subplots(len(indices), 6, figsize=(16, 3*len(indices)), squeeze=False)
    sx, sy = image.header.get_zooms()[:2]
    for v, z in enumerate(indices):
        for col, t in enumerate([0, 8, 15]):
            source_time = meta["temporal_source_indices"][t]
            nearest = int(np.floor(source_time + .5)) % image.shape[3]
            axes[v, col*2].imshow(native[v, nearest], cmap="gray", vmin=0, vmax=1, aspect=sy/sx)
            axes[v, col*2].set_title("Native z=%d t=%d" % (z, nearest))
            axes[v, col*2+1].imshow(cache[v, t], cmap="gray", vmin=0, vmax=1)
            axes[v, col*2+1].set_title("New t=%d (source %.2f)" % (t, source_time))
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle(row["patient_id"]+" | native SAX vs new cache; source display uses nearest frame")
    fig.tight_layout()
    display(fig)
    plt.close(fig)
    print("Source:", image.shape, "| cache:", cache.shape, "| native Z:", indices.tolist(), "|", status)
    print("No Z reslicing. Unknown ED is not inferred. Cache:", path)
    return meta
