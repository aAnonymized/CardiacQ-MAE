from contextlib import contextmanager
from time import perf_counter
import json
from pathlib import Path
import torch


@contextmanager
def timed(root, uid, view, stage, device=None):
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)
    print("[Export]", uid, "view", view, stage, flush=True)
    start, status = perf_counter(), "ok"
    try:
        yield
    except BaseException:
        status = "failed"
        raise
    finally:
        if device is not None and device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = perf_counter() - start
        path = Path(root) / "export_timings.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(patient_uid=uid, view=view, stage=stage, seconds=elapsed, status=status)) + "\n")
        print("[Export]", stage, "%.2fs" % elapsed, status, flush=True)
