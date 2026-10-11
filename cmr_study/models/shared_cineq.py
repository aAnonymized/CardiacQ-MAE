"""Shared cine encoder/query-decoder foundation for cineQ-MAE."""
from copy import deepcopy

import torch
from torch import nn

from .study import Backbone
from .maest import verify_upstream
from .shared_query import SharedQueryDecoder
from ..motion.geometry import local_points, patch_anchors


class SharedCineQ(nn.Module):
    """Own exactly one encoder and one shared reconstruction/motion decoder.

    Loss construction lives in SparsePathCineQ so the implemented objective has
    exactly three public terms: reconstruction, DIS (including Jacobian), DPCC.
    """

    architecture_version = "shared_query_ccpe_three_loss_v3"

    def __init__(self, settings, method):
        super().__init__()
        verify_upstream()
        self.settings, self.method = deepcopy(settings), method
        s, m = self.settings, self.settings["model"]
        if s["data"]["num_frames"] != 16 or m["norm_pix_loss"]:
            raise ValueError("Shared CineQ requires T=16 and raw-intensity targets")
        self.fraction = float(s["query"].get("query_fraction", .10))
        self.top_share = float(s["query"].get("top_share", .70))
        self.confidence_min = float(s["query"]["confidence_min"])
        self.dis_weight = float(s["query"].get("dis_loss_weight", 1.))
        self.jacobian_weight = float(s["query"].get("jacobian_loss_weight", .1))
        self.dpcc_weight = float(s.get("shared_decoder", {}).get("dpcc_loss_weight", .1))
        if not (0 < self.fraction <= 1 and 0 <= self.top_share <= 1 and
                0 <= self.confidence_min <= 1):
            raise ValueError("Invalid sparse query fractions or confidence threshold")
        if min(self.dis_weight, self.jacobian_weight, self.dpcc_weight) < 0:
            raise ValueError("Loss weights must be nonnegative")

        self.backbone = Backbone(s)
        # The original MAE decoder is removed, not merely bypassed.
        for name in list(self.backbone._modules) + list(self.backbone._parameters):
            if name.startswith("decoder") or name == "mask_token":
                delattr(self.backbone, name)
        self.query = SharedQueryDecoder(s)
        self.motion_decoder = None
        self.register_buffer("anchors", patch_anchors(s["data"]["input_size"],
                                                     m["patch_size"]), persistent=False)

    def reconstruction_parameter(self):
        return self.query.pixel_head.weight

    @staticmethod
    def _generator(device, seed, offset=0):
        return None if seed is None else torch.Generator(device=device).manual_seed(int(seed) + offset)

    def _center_tracks(self, prepared, spatial_ids, source_times):
        """Query each selected patch center at all 16 target phases."""
        b, q = spatial_ids.shape
        points = self.anchors[spatial_ids][:, :, None].expand(b, q, 16, 2).reshape(b, -1, 2)
        source = source_times[:, :, None].expand(b, q, 16).reshape(b, -1)
        target = torch.arange(16, device=points.device)[None, None].expand(
            b, q, 16).reshape(b, -1)
        return self.query.decode(prepared, points, source, target, 1).reshape(b, q, 16, 2)

    def _sampled_neighbors(self, prepared, spatial_ids, source_times, target_times):
        """Query four local neighbors only at the configured Jacobian phases."""
        b, q = spatial_ids.shape
        if target_times.ndim != 3 or target_times.shape[:2] != (b, q):
            raise ValueError("target_times must be [B,Q,S]")
        samples = target_times.shape[-1]
        local = local_points(self.anchors[spatial_ids].reshape(-1, 2),
                             self.settings["model"]["patch_size"]).reshape(b, q, 5, 2)
        neighbors = local[..., 1:, :]
        points = neighbors[:, :, None].expand(b, q, samples, 4, 2).reshape(b, -1, 2)
        source = source_times[:, :, None, None].expand(b, q, samples, 4).reshape(b, -1)
        target = target_times[:, :, :, None].expand(b, q, samples, 4).reshape(b, -1)
        return self.query.decode(prepared, points, source, target, 1).reshape(
            b, q, samples, 4, 2)

    def _tracks(self, prepared, spatial_ids, source_times):
        """Full five-point tracks retained for infrequent snapshot diagnostics."""
        b, q = spatial_ids.shape
        local = local_points(self.anchors[spatial_ids].reshape(-1, 2),
                             self.settings["model"]["patch_size"]).reshape(b, q, 5, 2)
        points = local[:, :, None].expand(b, q, 16, 5, 2).reshape(b, -1, 2)
        source = source_times[:, :, None, None].expand(b, q, 16, 5).reshape(b, -1)
        target = torch.arange(16, device=points.device)[None, None, :, None].expand(
            b, q, 16, 5).reshape(b, -1)
        return self.query.decode(prepared, points, source, target, 1).reshape(b, q, 16, 5, 2)

    def predict_tracks(self, context, spatial_ids, source_times):
        if context.shape[0] != 1 or spatial_ids.ndim != 1:
            raise ValueError("predict_tracks expects one sample and 1D spatial ids")
        return self._tracks(self.query.prepare(context), spatial_ids[None], source_times[None])[0]
