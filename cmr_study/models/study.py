"""CMR-controlled adaptations on the verified official MAE-ST reconstruction backbone."""
from contextlib import contextmanager
from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F
from .maest import GrayMAEST, verify_upstream
from .masking import make_mask
from .query import DirectQuery
from ..motion.geometry import patch_anchors, local_points, components, importance


class Backbone(GrayMAEST):
    def __init__(self, settings):
        super().__init__(settings)
        self.forced_mask = None

    @contextmanager
    def mask_context(self, mask):
        previous = self.forced_mask
        self.forced_mask = mask
        try:
            yield
        finally:
            self.forced_mask = previous

    def random_masking(self, x, ratio):
        if self.forced_mask is None:
            return super().random_masking(x, ratio)
        mask = self.forced_mask.to(x.device)
        counts = (mask == 0).sum(1)
        if not torch.all(counts == counts[0]):
            raise ValueError("Batch masks must have equal visible counts")
        order = torch.argsort(mask*x.shape[1]+torch.arange(x.shape[1], device=x.device)[None], dim=1)
        keep = order[:, :int(counts[0])]
        return x.gather(1, keep[:, :, None].expand(-1, -1, x.shape[-1])), mask, order.argsort(1), keep

    def features(self, video):
        mask = video.new_zeros(video.shape[0], self.patch_embed.num_patches)
        with self.mask_context(mask):
            return self.forward_encoder(video, 0)[0]


class MotionDecoder(nn.Module):
    def __init__(self, backbone, depth=2):
        super().__init__()
        self.project = deepcopy(backbone.decoder_embed)
        self.blocks = deepcopy(backbone.decoder_blocks[:depth])
        self.norm = deepcopy(backbone.decoder_norm)
        dim = backbone.decoder_pred.in_features
        self.token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, backbone.patch_embed.num_patches, dim))
        nn.init.trunc_normal_(self.pos, std=.02)
        self.head = nn.Linear(dim, 8)

    def forward(self, latent, restore):
        x = self.project(latent)
        x = torch.cat([x, self.token.expand(x.shape[0], restore.shape[1]-x.shape[1], -1)], 1)
        x = x.gather(1, restore[..., None].expand(-1, -1, x.shape[-1]))+self.pos
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))


def track_targets(positions, settings):
    # Official TrackMAE pixel variant: 2x spatial track interpolation, adjacent displacement,
    # repeat the last displacement, four local vectors per temporal-spatial token.
    b, _, p, _ = positions.shape
    g, u = int(p**.5), settings["model"]["tubelet_size"]
    coarse = positions[:, ::u].reshape(b, -1, g, g, 2).permute(0, 4, 1, 2, 3)
    dense = F.interpolate(coarse, size=(coarse.shape[2], 2*g, 2*g), mode="trilinear", align_corners=True)
    delta = dense[:, :, 1:]-dense[:, :, :-1]
    delta = torch.cat([delta, delta[:, :, -1:]], 2)
    delta = delta.reshape(b, 2, -1, g, 2, g, 2).permute(0, 2, 3, 5, 4, 6, 1).reshape(b, -1, 4, 2)
    if settings["trackmae"]["normalize_motion"] == "patch":
        delta = (delta-delta.mean(-2, keepdim=True))/(delta.var(-2, keepdim=True).sqrt()+1e-6)
    return delta.flatten(-2)


class LegacyStudyModel(nn.Module):
    def __init__(self, settings, method):
        super().__init__()
        verify_upstream()
        self.settings, self.method = deepcopy(settings), method
        self.backbone = Backbone(settings)
        self.query = DirectQuery(settings) if method.startswith("cineq") else None
        self.motion_decoder = MotionDecoder(self.backbone) if method == "trackmae" else None
        self.register_buffer("anchors", patch_anchors(settings["data"]["input_size"], settings["model"]["patch_size"]), persistent=False)

    def reconstruction_parameter(self):
        return self.backbone.decoder_pred.weight

    def predict_tracks(self, context, spatial_ids, source_times):
        """One sample. Query only specified spatial patches/source phases."""
        s, p = self.settings, self.settings["model"]["patch_size"]
        local = local_points(self.anchors[spatial_ids], p)
        count, frames = len(spatial_ids), s["data"]["num_frames"]
        points = local[:, None].expand(-1, frames, -1, -1).reshape(1, -1, 2)
        src = source_times[:, None, None].expand(-1, frames, 5).reshape(1, -1)
        dst = torch.arange(frames, device=context.device)[None, :, None].expand(count, -1, 5).reshape(1, -1)
        return self.query(context, points, src, dst).reshape(count, frames, 5, 2)

    def forward(self, batch, epoch=0, eval_seed=None, force_mask=None, auxiliary=True):
        video = batch["video"]
        s, m = self.settings, self.settings["model"]
        b, spatial, temporal = video.shape[0], len(self.anchors), 16//m["tubelet_size"]
        strategy = "random" if eval_seed is not None else "tube" if self.method == "videomae" else "motion" if self.method == "trackmae" else m["mask_strategy"]
        score = (batch["positions"][:, 1:]-batch["positions"][:, :-1]).norm(dim=-1).mean(1) if strategy == "motion" else None
        mask = force_mask if force_mask is not None else make_mask(b, temporal, spatial, m["mask_ratio"], video.device,
            strategy, eval_seed, score, s["trackmae"]["motion_visible_fraction"])
        with self.backbone.mask_context(mask):
            latent, actual, restore = self.backbone.forward_encoder(video, m["mask_ratio"])
        prediction = self.backbone.forward_decoder(latent, restore)
        raw = self.backbone.patchify(video.float())
        error_pixels = (prediction.float()-raw).square()
        error = error_pixels.mean(-1)
        weighted_error = error
        query_loss, track_loss = video.new_zeros(()), video.new_zeros(())
        query_errors, valid_counts = [], []
        queried = 0
        weight = torch.ones_like(error)
        if auxiliary and self.method == "heartmae":
            wm = batch["weight"][:, None, None].expand(-1, 1, 16, -1, -1)
            weighted_error = (error_pixels*(1+self.backbone.patchify(wm))).mean(-1)
        elif auxiliary and self.method == "trackmae":
            target = track_targets(batch["positions"].float(), s)
            out = self.motion_decoder(latent, restore).float()
            track_loss = (((out-target).square().mean(-1)*mask).sum(1)/mask.sum(1)).mean()
        elif auxiliary and (self.query is not None or self.method in ("mae_dis", "mae_registration")):
            # The full-video pass is only for Query; raw pixels/features never enter the MAE decoder.
            context = self.backbone.features(video) if self.query is not None else None
            losses = []
            component = {"cineq_k": "k", "cineq_r": "r", "cineq_uniform": "none"}.get(self.method, "full")
            for n in range(b):
                ids = mask[n].nonzero().flatten()
                sp, source = ids % spatial, (ids//spatial)*m["tubelet_size"]
                if self.query is not None:
                    tracks = self.predict_tracks(context[n:n+1], sp, source)
                    queried += len(ids)*16*5
                    truth = batch["positions"][n, ids//spatial, :, sp].float()
                    confidence = batch["confidence"][n, ids//spatial, :, sp].float()
                    confidence = confidence*(confidence >= s["query"]["confidence_min"])
                    difference = F.smooth_l1_loss(tracks/(video.shape[-1]-1), truth/(video.shape[-1]-1), reduction="none", beta=1/(video.shape[-1]-1)).mean(-1)
                    losses.append((difference*confidence).sum()/confidence.sum().clamp_min(1))
                    query_errors.append(((tracks-truth).norm(dim=-1)*confidence).sum().detach())
                    valid_counts.append(confidence.sum().detach())
                else:
                    tracks = batch["positions"][n, ids//spatial, :, sp].float()
                # Detached weights prevent Query from minimizing image loss by collapsing M.
                with torch.no_grad():
                    k, r = components(tracks.detach(), self.anchors[sp], m["patch_size"])
                    imp = torch.zeros_like(k)
                    for t in source.unique():
                        select = source == t
                        imp[select] = importance(k[select], r[select], s["weights"], component)
                    ramp = min(1., (epoch+1)/max(1, s["weights"]["ramp_epochs"]))
                    weight[n, ids] = 1+s["weights"]["alpha"]*ramp*imp
            query_loss = torch.stack(losses).mean() if losses else query_loss
            weighted_error = error*weight
        mae_loss = ((weighted_error*mask).sum(1)/mask.sum(1)).mean()
        mse = ((error*mask).sum(1)/mask.sum(1)).mean()
        visible_mean = ((raw.mean(-1)*(1-mask)).sum(1)/(1-mask).sum(1))
        baseline_error = (raw-visible_mean[:, None, None]).square().mean(-1)
        baseline = ((baseline_error*mask).sum(1)/mask.sum(1)).mean()
        loss = mae_loss+s["query"]["loss_weight"]*query_loss+s["trackmae"]["loss_weight"]*track_loss
        metrics = dict(loss=loss, masked_mse=mse, mean_fill_mse=baseline, weighted_mse=mae_loss,
                       query_loss=query_loss, track_loss=track_loss, query_points=queried,
                       query_dis_epe_px=sum(query_errors)/torch.stack(valid_counts).sum().clamp_min(1) if valid_counts else 0.)
        return dict(loss=loss, prediction=prediction, mask=mask, weight=weight, metrics=metrics)


def StudyModel(settings, method):
    """Keep the public constructor; only cineQ methods use the shared decoder."""
    if method.startswith("cineq"):
        from .sparse_paths import SparsePathCineQ
        return SparsePathCineQ(settings, method)
    return LegacyStudyModel(settings, method)
