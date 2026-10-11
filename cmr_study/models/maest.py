"""Thin grayscale adapter over unmodified, pinned upstream MAE-ST."""
import hashlib
import json
import sys
from contextlib import contextmanager
from functools import partial
from pathlib import Path

import torch
from torch import nn

VENDOR = Path(__file__).resolve().parents[2] / "vendor"
sys.path.insert(0, str(VENDOR))
from mae_st.models_mae import MaskedAutoencoderViT


def verify_upstream():
    import mae_st.models_mae as upstream
    root = VENDOR / "mae_st"
    if Path(upstream.__file__).resolve() != (root / "models_mae.py").resolve():
        raise RuntimeError("Another mae_st is already imported. Restart the kernel and use this package.")
    manifest = json.loads((root / "source_manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["sha256"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError("Pinned upstream source was modified: " + name)
    return manifest


class GrayMAEST(MaskedAutoencoderViT):
    def __init__(self, settings):
        m, d = settings["model"], settings["data"]
        self.mask_strategy = m["mask_strategy"]
        self.mask_seed = None
        self.gray_size, self.gray_frames = d["input_size"], d["num_frames"]
        super().__init__(img_size=d["input_size"], patch_size=m["patch_size"], in_chans=1,
                         embed_dim=m["embed_dim"], depth=m["depth"], num_heads=m["num_heads"],
                         decoder_embed_dim=m["decoder_dim"], decoder_depth=m["decoder_depth"],
                         decoder_num_heads=m["decoder_heads"], norm_layer=partial(nn.LayerNorm, eps=1e-6),
                         num_frames=d["num_frames"], t_patch_size=m["tubelet_size"],
                         pred_t_dim=d["num_frames"], sep_pos_embed=True, cls_embed=True,
                         norm_pix_loss=m["norm_pix_loss"])

    def patchify(self, imgs):
        b, c, t, h, w = imgs.shape
        p, u = self.patch_embed.patch_size[0], self.t_pred_patch_size
        if c != 1 or h != w or h % p or t % u:
            raise ValueError("Expected B,1,T,H,W with integral patch dimensions")
        x = imgs.reshape(b, c, t // u, u, h // p, p, w // p, p)
        return torch.einsum("nctuhpwq->nthwupqc", x).reshape(b, -1, u * p * p * c)

    def unpatchify(self, x):
        b = x.shape[0]
        p, u = self.patch_embed.patch_size[0], self.t_pred_patch_size
        t, h = self.gray_frames // u, self.gray_size // p
        x = x.reshape(b, t, h, h, u, p, p, 1)
        return torch.einsum("nthwupqc->nctuhpwq", x).reshape(b, 1, self.gray_frames, self.gray_size, self.gray_size)

    def forward_loss(self, imgs, pred, mask):
        # Keep the upstream target/loss definition; accumulate MSE in FP32 under AMP.
        return super().forward_loss(imgs.float(), pred.float(), mask)

    @contextmanager
    def fixed_mask(self, seed):
        previous = self.mask_seed
        self.mask_seed = seed
        try:
            yield
        finally:
            self.mask_seed = previous

    def random_masking(self, x, mask_ratio):
        if self.mask_seed is None and self.mask_strategy == "random":
            return super().random_masking(x, mask_ratio)
        b, length, dim = x.shape
        generator = None
        if self.mask_seed is not None:
            generator = torch.Generator(device=x.device).manual_seed(int(self.mask_seed))
        if self.mask_strategy == "random":
            order = torch.rand(b, length, device=x.device, generator=generator).argsort(1)
            keep = int(length * (1 - mask_ratio))
        else:
            t, h, w = self.input_size
            spatial = h * w
            nkeep = max(1, min(spatial - 1, int(spatial * (1 - mask_ratio))))
            base = torch.ones(b, spatial, device=x.device)
            selected = torch.rand(b, spatial, device=x.device, generator=generator).argsort(1)[:, :nkeep]
            base.scatter_(1, selected, 0)
            tube = base[:, None].expand(-1, t, -1).reshape(b, length)
            order = torch.argsort(tube * length + torch.arange(length, device=x.device)[None], dim=1)
            keep = t * nkeep
        restore = order.argsort(1)
        kept = order[:, :keep]
        visible = torch.gather(x, 1, kept[:, :, None].expand(-1, -1, dim))
        mask = torch.ones(b, length, device=x.device)
        mask[:, :keep] = 0
        return visible, torch.gather(mask, 1, restore), restore, kept


def build_model(settings):
    verify_upstream()
    return GrayMAEST(settings)


def target_tokens(model, video):
    raw = model.patchify(video.float())
    mean = raw.mean(-1, keepdim=True)
    std = (raw.var(-1, keepdim=True) + 1e-6).sqrt()
    return raw, (raw - mean) / std if model.norm_pix_loss else raw, mean, std


def reconstruction_metrics(model, video, prediction, mask):
    raw, target, mean, std = target_tokens(model, video)
    prediction = prediction.float()
    per_patch = (prediction - target).square().mean(-1)
    denom = mask.sum(-1)
    mse = ((per_patch * mask).sum(-1) / denom).mean()
    # No masked target values enter the raw-mode baseline.
    visible_mean = (raw.mean(-1) * (1 - mask)).sum(-1) / (1 - mask).sum(-1)
    baseline = torch.zeros_like(target) if model.norm_pix_loss else visible_mean[:, None, None].expand_as(target)
    baseline_mse = (((baseline - target).square().mean(-1) * mask).sum(-1) / denom).mean()
    pixels = model.unpatchify(mask[:, :, None].expand_as(prediction)).bool()
    masked_pred = prediction[mask.bool()]
    metrics = {"masked_mse": float(mse.detach()), "baseline_mse": float(baseline_mse.detach()),
               "relative_gain": float((1 - mse / baseline_mse.clamp_min(1e-12)).detach()),
               "masked_prediction_std": float(masked_pred.std().detach()),
               "masked_prediction_min": float(masked_pred.min().detach()),
               "masked_prediction_max": float(masked_pred.max().detach())}
    if not model.norm_pix_loss:
        metrics["masked_psnr_db"] = float((-10 * mse.clamp_min(1e-12).log10()).detach())
    else:
        # This is a target-statistics-assisted display, NOT raw-intensity prediction.
        restored = prediction * std + mean
        metrics["oracle_denorm_masked_mse"] = float((((restored - raw).square().mean(-1) * mask).sum(-1) / denom).mean().detach())
    return metrics, pixels
