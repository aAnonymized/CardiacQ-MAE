import math
from contextlib import nullcontext
from itertools import islice
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from ..utils import case_seed, write_json


def device_for(settings):
    name = settings["device"]
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else "cpu" if name == "auto" else name)


def make_scaler(enabled):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled, init_scale=1024.)
    return torch.cuda.amp.GradScaler(enabled=enabled, init_scale=1024.)


def optimizer_for(model, config):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or "pos" in name or "token" in name else decay).append(p)
    return torch.optim.AdamW([dict(params=decay, weight_decay=config["weight_decay"]),
                             dict(params=no_decay, weight_decay=0.)], lr=config["lr"], betas=(.9, .95))


def lr_at(config, epoch):
    warm = config["warmup_epochs"]
    if warm and epoch < warm:
        return config["lr"]*epoch/warm
    p = min(1., max(0., (epoch-warm)/max(1, config["epochs"]-warm)))
    return config["min_lr"]+(config["lr"]-config["min_lr"])*.5*(1+math.cos(math.pi*p))


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def loader(dataset, s, kind):
    return DataLoader(dataset, batch_size=s[kind]["batch_size"], shuffle=True, drop_last=False,
                      num_workers=s["num_workers"], pin_memory=torch.cuda.is_available(), persistent_workers=s["num_workers"] > 0)


def evaluation_loader(dataset, device):
    device = torch.device(device)
    options = dataset.settings["evaluation"]
    return DataLoader(dataset, batch_size=options["batch_size"], shuffle=False, drop_last=False,
                      num_workers=options["num_workers"], pin_memory=device.type == "cuda")


def finish_update(model, optimizer, scaler, max_norm):
    """Never clip nonfinite gradients or let them reach an optimizer update."""
    scaler.unscale_(optimizer)
    named = [(name, p) for name, p in model.named_parameters() if p.grad is not None]
    if not named:
        raise RuntimeError("No gradients: check the trainable parameters and loss graph")
    # FP64 accumulation avoids overflowing the norm of finite FP32 gradients.
    norm = torch.linalg.vector_norm(torch.stack([
        torch.linalg.vector_norm(p.grad.detach(), dtype=torch.float64) for _, p in named]))
    scale = scaler.get_scale()
    finite = bool(torch.isfinite(norm))
    bad = [name for name, p in named if not torch.isfinite(p.grad).all()] if not finite else []
    if finite:
        coefficient = (max_norm/(norm+1e-6)).clamp(max=1.)
        for _, p in named:
            p.grad.mul_(coefficient.to(device=p.grad.device, dtype=p.grad.dtype))
        scaler.step(optimizer)
        scaler.update()
    elif scaler.is_enabled():
        # unscale_ has recorded the overflow; GradScaler skips optimizer.step.
        scaler.step(optimizer)
        scaler.update()
    optimizer.zero_grad(set_to_none=True)
    return dict(skipped=not finite, grad_norm=float(norm) if finite else None,
                bad_parameters=bad[:20], amp_scale_before=scale, amp_scale_after=scaler.get_scale())


def epoch_train(model, batches, optimizer, scaler, config, device, epoch, forward, diagnostic_dir=None):
    model.train()
    steps = math.ceil(len(batches)/config["accum_steps"])
    it = iter(batches)
    sums, count, skips, norms, consecutive, updates = {}, 0, 0, [], 0, 0
    for step in tqdm(range(steps), desc="Epoch %d" % (epoch+1), leave=False):
        group = list(islice(it, config["accum_steps"]))
        patients = [uid for batch in group for uid in batch.get("patient_uid", [])]
        def report_issue(reason, **extra):
            if diagnostic_dir is not None:
                from pathlib import Path
                write_json(Path(diagnostic_dir)/("numerical_issue_epoch%04d_step%05d.json" % (epoch+1, step+1)),
                           dict(reason=reason, epoch=epoch+1, step=step+1, patient_uids=patients,
                                amp=scaler.is_enabled(), lr=rate, **extra))
        n = sum(len(x["video"]) for x in group)
        optimizer.zero_grad(set_to_none=True)
        rate = lr_at(config, epoch+(step+1)/steps)
        for pg in optimizer.param_groups:
            pg["lr"] = rate
        for batch in group:
            batch = to_device(batch, device)
            ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if scaler.is_enabled() else nullcontext()
            try:
                with ctx:
                    loss, metrics = forward(model, batch, epoch)
            except (FloatingPointError, ValueError) as exc:
                optimizer.zero_grad(set_to_none=True)
                report_issue(str(exc))
                raise
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                detail = {k: float(v.detach().float().cpu()) if torch.is_tensor(v) else float(v) for k, v in metrics.items()}
                ranges = {k: dict(shape=list(v.shape), finite=bool(torch.isfinite(v).all()),
                                  minimum=float(v.min()), maximum=float(v.max()))
                          for k, v in batch.items() if torch.is_tensor(v) and v.is_floating_point()}
                report_issue("Nonfinite forward loss", metrics=detail, tensors=ranges)
                raise FloatingPointError("Nonfinite loss for %s. Last completed checkpoint retained." % patients)
            size = len(batch["video"])
            scaler.scale(loss*size/n).backward()
            for k, value in metrics.items():
                value = float(value.detach()) if torch.is_tensor(value) else float(value)
                sums[k] = sums.get(k, 0.)+value*size
            count += size
        update = finish_update(model, optimizer, scaler, config["grad_clip"])
        if update["skipped"]:
            skips += 1
            consecutive += 1
            report_issue("Nonfinite gradients; NO optimizer update", **update)
            message = "Nonfinite gradients at epoch %d step %d; patients=%s; parameters=%s" % (
                epoch+1, step+1, patients, update["bad_parameters"])
            if not scaler.is_enabled() or consecutive >= config.get("max_consecutive_amp_skips", 8):
                raise FloatingPointError(message + ". Stopped; inspect numerical_issue report. No checkpoint overwritten.")
            print(message + "; AMP skipped update and reduced scale to " + str(update["amp_scale_after"]), flush=True)
        else:
            consecutive = 0
            updates += 1
            norms.append(update["grad_norm"])
    if updates == 0:
        raise FloatingPointError("All optimizer updates skipped; do not save this as a trained epoch. Inspect numerical_issue reports.")
    return dict({k: v/count for k, v in sums.items()}, lr=rate, grad_norm=float(np.mean(norms)), amp_skipped=skips, optimizer_updates=updates)


def pretrain_forward(model, batch, epoch):
    out = model(batch, epoch=epoch)
    return out["loss"], out["metrics"]


@torch.no_grad()
def evaluate_reconstruction(model, dataset, seed, device, epoch=0):
    import pandas as pd
    model.eval()
    rows = []
    preferred = [
        "masked_mse", "mean_fill_mse", "weighted_mse",
        "reconstruction_loss", "dis_loss", "dis_coordinate_loss",
        "dis_jacobian_loss", "query_dis_epe_px", "dpcc_loss",
        # Legacy baselines retain these names.
        "query_loss", "track_loss",
    ]
    for item in tqdm(dataset, desc="ALL validation patients/views", leave=False):
        batch = {k: v[None].to(device) for k,v in item.items() if torch.is_tensor(v)}
        out = model(batch, epoch=epoch, eval_seed=case_seed(item["patient_uid"], item["view"], seed), auxiliary=True)
        keys = [key for key in preferred if key in out["metrics"]]
        rows.append(dict(patient_uid=item["patient_uid"], view=item["view"],
                         **{k: float(out["metrics"][k]) for k in keys}))
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("Empty validation dataset")
    keys = [key for key in preferred if key in frame.columns]
    values = frame.groupby("patient_uid")[keys].mean().mean().to_dict()
    values.update(relative_gain=1-values["masked_mse"]/max(values["mean_fill_mse"], 1e-12),
                  patients=frame.patient_uid.nunique(), views=len(frame))
    return values, frame
