"""Full-cycle cineQ-MAE with three losses: reconstruction, DIS, and DPCC."""
import math
import time
from contextlib import contextmanager

import torch
from torch.nn import functional as F

from .shared_cineq import SharedCineQ
from .masking import make_mask
from ..motion.geometry import importance, jacobians, local_points


class SparsePathCineQ(SharedCineQ):
    architecture_version = "shared_fullcycle_center_sampled_jacobian_three_loss_v4"

    def __init__(self, settings, method):
        super().__init__(settings, method)
        options = self.settings.get("sparse_paths", {})
        self.phase_split = float(options.get("phase_split_fraction", 2. / 3.))
        self.full_cycle_weight = float(options.get(
            "full_cycle_weight", options.get("first_hop_weight", .5)))
        self.composed_endpoint_weight = float(options.get(
            "composed_endpoint_weight", options.get("endpoint_weight", .5)))
        self.jacobian_phase_offsets = tuple(int(x) for x in options.get(
            "jacobian_phase_offsets", [2, 6, 10, 14]))
        self.jacobian_beta = float(self.settings["query"].get("jacobian_beta", .1))
        if not isinstance(self.settings["weights"].get("add_baseline", True), bool):
            raise ValueError("weights.add_baseline must be True or False")
        if not 0 < self.phase_split < 1:
            raise ValueError("phase_split_fraction must be between 0 and 1")
        cut = math.floor(16 * self.phase_split)
        if min(cut + 1, 15 - cut) < 3:
            raise ValueError("Each phase bin needs >=3 phases before excluding source/target")
        groups = (self.full_cycle_weight, self.composed_endpoint_weight)
        if not all(math.isfinite(x) and x >= 0 for x in groups) or sum(groups) <= 0:
            raise ValueError("DIS group weights must be finite/nonnegative with positive sum")
        if not math.isfinite(self.jacobian_beta) or self.jacobian_beta <= 0:
            raise ValueError("query.jacobian_beta must be positive")
        if (len(self.jacobian_phase_offsets) != 4 or
                len(set(self.jacobian_phase_offsets)) != 4 or
                any(x <= 0 or x >= 16 for x in self.jacobian_phase_offsets)):
            raise ValueError("sparse_paths.jacobian_phase_offsets needs four distinct values in [1,15]")

    @staticmethod
    @contextmanager
    def _timed(name, video, timings, enabled):
        if not enabled:
            yield
            return
        if video.is_cuda:
            torch.cuda.synchronize(video.device)
        start = time.perf_counter()
        with torch.autograd.profiler.record_function("cineq/" + name):
            yield
        if video.is_cuda:
            torch.cuda.synchronize(video.device)
        timings[name + "_ms"] = 1000 * (time.perf_counter() - start)

    @torch.no_grad()
    def _sample_plan(self, sources, ids, valid, batch, eval_seed):
        b, q = sources.shape
        generator = self._generator(sources.device, eval_seed, 104729)
        supplied = batch.get("target_phases")
        if supplied is None:
            target = torch.randint(0, 15, (b, q), device=sources.device, generator=generator)
            target = target + (target >= sources).long()
        else:
            supplied = torch.as_tensor(supplied, device=sources.device)
            expected = (b, len(self.anchors) * (16 // self.settings["model"]["tubelet_size"]))
            if tuple(supplied.shape) != expected:
                raise ValueError("batch.target_phases must be [B, number_of_tubelets]")
            target = supplied.gather(1, ids)
            bad = (~torch.isfinite(target) | (target != target.long()) |
                   (target < 0) | (target >= 16) | (target == sources)) & valid
            if bad.any():
                raise ValueError("Selected target phases must be integers in [0,15], different from source")
            target = torch.where(valid, target.long(), (sources + 1) % 16)

        phases = torch.arange(16, device=sources.device)
        eligible = (phases != sources[..., None]) & (phases != target[..., None])
        bins = (phases <= 16 * self.phase_split, phases > 16 * self.phase_split)
        mids = []
        for interval in bins:
            choices = eligible & interval
            noise = torch.rand(b, q, 16, device=sources.device, generator=generator)
            mids.append(noise.masked_fill(~choices, -1).argmax(-1))
        return target, torch.stack(mids, 2)

    @torch.no_grad()
    def select_queries(self, batch, mask, eval_seed=None):
        """Select Top70% + Random30% only among reliable masked tubelets."""
        b, spatial = len(mask), len(self.anchors)
        tube = self.settings["model"]["tubelet_size"]
        temporal = 16 // tube
        masked = mask.reshape(b, temporal, spatial).bool()
        centers = batch["positions"][..., 0, :].float()
        scores = (centers - self.anchors[None, None, None]).norm(dim=-1).mean(2)
        confidence = batch["confidence"][..., 0].float()
        other = (torch.arange(16, device=mask.device)[None, :] !=
                 torch.arange(temporal, device=mask.device)[:, None] * tube)
        mean_conf = (confidence * other[None, :, :, None]).sum(2) / 15
        reliable = (masked & torch.isfinite(scores) & torch.isfinite(mean_conf) &
                    (mean_conf >= self.confidence_min))
        budget = torch.ceil(masked.sum(-1).float() * self.fraction).long()
        budget = torch.minimum(budget, reliable.sum(-1))
        top_budget = torch.ceil(budget.float() * self.top_share).long()
        rank = scores.masked_fill(~reliable, -torch.inf).argsort(-1, descending=True).argsort(-1)
        top = reliable & (rank < top_budget[..., None])
        remaining = reliable & ~top
        noise = torch.rand(scores.shape, device=mask.device,
                           generator=self._generator(mask.device, eval_seed, 1009))
        random_rank = noise.masked_fill(~remaining, torch.inf).argsort(-1).argsort(-1)
        selected = (top | (remaining &
                    (random_rank < (budget - top_budget)[..., None]))).flatten(1)
        maximum = int(selected.sum(1).max())
        index = torch.arange(mask.shape[1], device=mask.device)[None].expand_as(mask)
        order = index.masked_fill(~selected, mask.shape[1]).argsort(1)[:, :maximum]
        valid = selected.gather(1, order)
        return torch.where(valid, order, torch.zeros_like(order)), valid

    @staticmethod
    def _teacher_at(batch, ids, times, spatial):
        rows = torch.arange(ids.shape[0], device=ids.device)[:, None, None]
        source = (ids // spatial)[..., None]
        patch = (ids % spatial)[..., None]
        positions = batch["positions"][rows, source, times, patch].float()
        confidence = batch["confidence"][rows, source, times, patch].float()
        if not torch.isfinite(positions).all() or not torch.isfinite(confidence).all():
            raise FloatingPointError("Nonfinite DIS teacher; rebuild the affected motion cache")
        return positions, confidence.clamp(0, 1)

    def _coordinate_loss(self, predicted, truth, confidence):
        """Smooth-L1 in patch units, with EPE retained in pixels for reporting."""
        patch = float(self.settings["model"]["patch_size"])
        error = F.smooth_l1_loss(predicted, truth, reduction="none", beta=1.).mean(-1) / patch
        dims = tuple(range(1, confidence.ndim))
        denominator = confidence.sum(dims).clamp_min(1)
        loss = ((error * confidence).sum(dims) / denominator).mean()
        epe = (((predicted - truth).norm(dim=-1) * confidence).sum(dims) /
               denominator).mean()
        return loss, epe.detach()

    def _jacobian_loss(self, predicted, truth, confidence, valid):
        """Local Query-DIS geometry; this is an internal part of L_DIS."""
        patch = self.settings["model"]["patch_size"]
        predicted_j = jacobians(predicted, patch)
        truth_j = jacobians(truth, patch)
        reliability = confidence.amin(-1) * valid[..., None]
        error = F.smooth_l1_loss(predicted_j, truth_j, reduction="none",
                                 beta=self.jacobian_beta).mean((-1, -2))
        denominator = reliability.sum((1, 2)).clamp_min(1)
        loss = ((error * reliability).sum((1, 2)) / denominator).mean()
        frobenius = ((predicted_j - truth_j).square().sum((-1, -2)).sqrt() * reliability)
        metric = (frobenius.sum((1, 2)) / denominator).mean()
        return loss, metric.detach()

    @torch.no_grad()
    def _trajectory_weights(self, center_tracks, sampled_local_tracks,
                            center_confidence, local_confidence,
                            ids, valid, error, epoch):
        """TGWR with full-cycle center K and sampled-phase local Jacobian R."""
        options = self.settings["weights"]
        patch = self.settings["model"]["patch_size"]
        spatial = len(self.anchors)
        center_tracks = center_tracks.detach().float()
        sampled_local_tracks = sampled_local_tracks.detach().float()
        anchors = self.anchors[ids % spatial]
        k_phase = (center_tracks - anchors[:, :, None]).norm(dim=-1)
        identity = torch.eye(2, device=center_tracks.device, dtype=center_tracks.dtype)
        r_phase = (jacobians(sampled_local_tracks, patch) - identity).square().sum(
            (-1, -2)).sqrt()

        usable_k = ((center_confidence >= self.confidence_min) & valid[..., None])
        usable_r = ((local_confidence.amin(-1) >= self.confidence_min) & valid[..., None])
        k_weights = usable_k.float() / usable_k.sum(-1, keepdim=True).clamp_min(1)
        r_weights = usable_r.float() / usable_r.sum(-1, keepdim=True).clamp_min(1)
        k = (k_phase * k_weights).sum(-1)
        r = (r_phase * r_weights).sum(-1)
        active_k = valid & usable_k.any(-1)
        active_r = valid & usable_r.any(-1)
        component = {"cineq_k": "k", "cineq_r": "r",
                     "cineq_uniform": "none"}.get(self.method, "full")
        active = (active_k if component == "k" else active_r if component == "r"
                  else valid if component == "none" else active_k & active_r)
        motion_importance = importance(k, r, options, component)
        ramp = min(1., (epoch + 1) / max(1, options["ramp_epochs"]))
        baseline = 1. if options.get("add_baseline", True) else 0.
        desired = baseline + options["alpha"] * ramp * motion_importance
        if component == "none":
            desired = torch.ones_like(desired)

        result = torch.ones_like(error)
        delta = torch.where(active, desired - 1., torch.zeros_like(desired))
        result.scatter_add_(1, ids, delta)
        return result, dict(full_cycle_k=k, full_cycle_r=r, tgwr_valid=active,
                            k_phase_weights=k_weights,
                            r_phase_weights=r_weights,
                            tgwr_weights=result)

    def _paths_and_losses(self, prepared, batch, ids, valid, raw, error,
                          epoch, eval_seed, timings, profile):
        video, settings = batch["video"], self.settings
        b, q = ids.shape
        spatial = len(self.anchors)
        tube = settings["model"]["tubelet_size"]
        sources = (ids // spatial) * tube
        target, mids = self._sample_plan(sources, ids, valid, batch, eval_seed)
        offsets = torch.as_tensor(self.jacobian_phase_offsets, device=ids.device)
        jacobian_phases = (sources[..., None] + offsets) % 16

        with self._timed("full_cycle_tracks", video, timings, profile):
            center_tracks = self._center_tracks(prepared, ids % spatial, sources)

        with self._timed("sampled_jacobian_neighbors", video, timings, profile):
            neighbor_tracks = self._sampled_neighbors(
                prepared, ids % spatial, sources, jacobian_phases)

        rows3 = torch.arange(b, device=ids.device)[:, None, None]
        cols3 = torch.arange(q, device=ids.device)[None, :, None]
        rows2 = torch.arange(b, device=ids.device)[:, None]
        cols2 = torch.arange(q, device=ids.device)[None, :]
        sampled_centers = center_tracks[rows3, cols3, jacobian_phases]
        sampled_local_tracks = torch.cat(
            [sampled_centers[..., None, :], neighbor_tracks], dim=-2)
        first_centers = center_tracks[rows3, cols3, mids]
        direct = center_tracks[rows2, cols2, target]

        with self._timed("composed_endpoints", video, timings, profile):
            second = self.query.decode(
                prepared,
                first_centers.reshape(b, -1, 2),
                mids.reshape(b, -1),
                target[..., None].expand(b, q, 2).reshape(b, -1),
                1,
            ).reshape(b, q, 2, 2)

        with self._timed("dis_and_tgwr", video, timings, profile):
            all_times = torch.arange(16, device=ids.device)[None, None].expand(b, q, 16)
            teacher, confidence = self._teacher_at(batch, ids, all_times, spatial)
            confidence = (confidence * (confidence >= self.confidence_min) *
                          valid[:, :, None, None])

            teacher_centers = teacher[..., 0, :]
            center_confidence = confidence[..., 0]
            sampled_teacher = teacher[rows3, cols3, jacobian_phases]
            sampled_confidence = confidence[rows3, cols3, jacobian_phases]
            center_loss, center_epe = self._coordinate_loss(
                center_tracks, teacher_centers, center_confidence)
            neighbor_loss, neighbor_epe = self._coordinate_loss(
                neighbor_tracks, sampled_teacher[..., 1:, :],
                sampled_confidence[..., 1:])
            # Sixteen center and sixteen neighbor queries contribute equally.
            full_loss = .5 * (center_loss + neighbor_loss)
            full_epe = .5 * (center_epe + neighbor_epe)
            target_truth = teacher_centers[rows2, cols2, target]
            target_confidence = center_confidence[rows2, cols2, target]
            mid_confidence = center_confidence[rows3, cols3, mids]
            endpoint_confidence = mid_confidence * target_confidence[..., None]
            endpoint_truth = target_truth[:, :, None].expand_as(second)
            endpoint_loss, endpoint_epe = self._coordinate_loss(
                second, endpoint_truth, endpoint_confidence)

            group_total = self.full_cycle_weight + self.composed_endpoint_weight
            coordinate_loss = ((self.full_cycle_weight * full_loss +
                               self.composed_endpoint_weight * endpoint_loss) / group_total)
            jacobian_loss, jacobian_error = self._jacobian_loss(
                sampled_local_tracks, sampled_teacher, sampled_confidence, valid)
            dis_loss = coordinate_loss + self.jacobian_weight * jacobian_loss
            weight, weight_debug = self._trajectory_weights(
                center_tracks, sampled_local_tracks,
                center_confidence, sampled_confidence,
                ids, valid, error, epoch)

        dpcc = video.new_zeros(())
        appearance_count = 0
        with self._timed("dpcc_three_paths", video, timings, profile):
            endpoint_tensor = torch.cat([direct[:, :, None], second], 2)
            app_confidence = torch.cat(
                [target_confidence[:, :, None], endpoint_confidence], 2) * valid[:, :, None]
            if self.dpcc_weight > 0:
                read_times = target[..., None].expand(b, q, 3).reshape(b, -1)
                appearance = self.query.decode(
                    prepared, endpoint_tensor.reshape(b, -1, 2),
                    read_times, read_times, 0).reshape(b, q, 3, -1)
                plane = settings["model"]["patch_size"] ** 2
                appearance = appearance[..., :plane]
                source_patch = raw.gather(
                    1, ids[..., None].expand(-1, -1, raw.shape[-1]))[..., :plane]
                appearance_error = (appearance - source_patch[:, :, None]).square().mean(-1)
                denominator = app_confidence.sum((1, 2)).clamp_min(1)
                dpcc = ((appearance_error * app_confidence).sum((1, 2)) /
                        denominator).mean()
                appearance_count = q * 3

        debug = dict(
            selected_ids=ids, selected_valid=valid, source_phases=sources,
            target_phases=target, mid_phases=mids,
            jacobian_phases=jacobian_phases,
            center_tracks=center_tracks,
            sampled_neighbor_tracks=neighbor_tracks,
            sampled_local_tracks=sampled_local_tracks,
            first_hops=first_centers, direct_endpoint=direct,
            endpoints=second, teacher_full_cycle_center=teacher_centers,
            teacher_sampled_local=sampled_teacher,
            teacher_target=target_truth, endpoint_confidence=endpoint_confidence,
            **weight_debug,
        )
        stats = dict(
            dis_loss=dis_loss,
            dis_coordinate_loss=coordinate_loss,
            dis_full_cycle_loss=full_loss,
            dis_center_full_cycle_loss=center_loss,
            dis_sampled_neighbor_loss=neighbor_loss,
            dis_endpoint_loss=endpoint_loss,
            dis_jacobian_loss=jacobian_loss,
            query_dis_epe_px=full_epe,
            dis_endpoint_epe_px=endpoint_epe,
            jacobian_error=jacobian_error,
            dpcc_loss=dpcc,
        )
        # Full center cycle: 16; four neighbors at four phases: 16; endpoints: 2.
        return weight, stats, debug, q * 34, appearance_count

    def forward(self, batch, epoch=0, eval_seed=None, force_mask=None,
                auxiliary=True, return_debug=False, profile=False):
        settings, model_cfg, video = self.settings, self.settings["model"], batch["video"]
        b = len(video)
        spatial = len(self.anchors)
        temporal = 16 // model_cfg["tubelet_size"]
        timings = {}

        with self._timed("mask_and_encode", video, timings, profile):
            mask = force_mask if force_mask is not None else make_mask(
                b, temporal, spatial, model_cfg["mask_ratio"], video.device,
                "random" if eval_seed is not None else model_cfg["mask_strategy"], eval_seed)
            if mask.shape != (b, spatial * temporal) or not ((mask == 0) | (mask == 1)).all():
                raise ValueError("Invalid binary token mask")
            if (mask.sum(1) == 0).any() or (mask.sum(1) == mask.shape[1]).any():
                raise ValueError("Both visible and masked tokens are required")
            with self.backbone.mask_context(mask):
                context, _, _ = self.backbone.forward_encoder(video, model_cfg["mask_ratio"])

        with self._timed("prepare_kv", video, timings, profile):
            prepared = self.query.prepare(context)

        with self._timed("masked_reconstruction", video, timings, profile):
            raw = self.backbone.patchify(video.float())
            if self.training and auxiliary:
                rec_ids = torch.stack([row.nonzero().flatten() for row in mask])
            else:
                rec_ids = torch.arange(mask.shape[1], device=video.device)[None].expand(b, -1)
            points = self.anchors[rec_ids % spatial]
            times = (rec_ids // spatial) * model_cfg["tubelet_size"]
            predicted = self.query.decode(prepared, points, times, times, 0)
            prediction = raw.new_zeros(raw.shape).scatter(
                1, rec_ids[..., None].expand_as(predicted), predicted.float())
            error = (prediction - raw).square().mean(-1)

        zero_names = (
            "dis_loss", "dis_coordinate_loss", "dis_full_cycle_loss",
            "dis_center_full_cycle_loss", "dis_sampled_neighbor_loss",
            "dis_endpoint_loss", "dis_jacobian_loss", "query_dis_epe_px",
            "dis_endpoint_epe_px", "jacobian_error", "dpcc_loss",
        )
        stats = {name: video.new_zeros(()) for name in zero_names}
        weight, debug = torch.ones_like(error), {}
        ids = torch.empty(b, 0, device=video.device, dtype=torch.long)
        valid = torch.zeros_like(ids, dtype=torch.bool)
        coordinate_count = appearance_count = 0

        if auxiliary:
            if "positions" not in batch or "confidence" not in batch:
                raise KeyError("DIS trajectory cache required; run prepare_motion()")
            with self._timed("select_anchors", video, timings, profile):
                ids, valid = self.select_queries(batch, mask, eval_seed)
            if ids.shape[1]:
                weight, stats, debug, coordinate_count, appearance_count = self._paths_and_losses(
                    prepared, batch, ids, valid, raw, error, epoch,
                    eval_seed, timings, profile)

        with self._timed("three_loss_objective", video, timings, profile):
            denominator = mask.sum(1)
            masked_mse = ((error * mask).sum(1) / denominator).mean()
            reconstruction_loss = ((error * weight * mask).sum(1) / denominator).mean()
            mean = (raw.mean(-1) * (1 - mask)).sum(1) / (1 - mask).sum(1)
            baseline = (((raw - mean[:, None, None]).square().mean(-1) * mask).sum(1) /
                        denominator).mean()
            # Exactly three externally weighted terms.
            loss = (reconstruction_loss + self.dis_weight * stats["dis_loss"] +
                    self.dpcc_weight * stats["dpcc_loss"])

        metrics = dict(
            loss=loss,
            reconstruction_loss=reconstruction_loss,
            dis_loss=stats["dis_loss"],
            dpcc_loss=stats["dpcc_loss"],
            masked_mse=masked_mse,
            mean_fill_mse=baseline,
            weighted_mse=reconstruction_loss,
            selected_anchors=valid.sum().detach(),
            query_points=valid.sum().detach() * 34,
            coordinate_queries_per_sample=coordinate_count,
            appearance_queries_per_sample=appearance_count,
            reconstruction_queries_per_sample=rec_ids.shape[1],
            total_queries_per_sample=coordinate_count + appearance_count + rec_ids.shape[1],
            mean_masked_weight=((weight * mask).sum(1) / denominator).mean().detach(),
            **{k: v for k, v in stats.items() if k not in ("dis_loss", "dpcc_loss")},
        )
        output = dict(loss=loss, prediction=prediction, mask=mask,
                      weight=weight, metrics=metrics)
        if not self.training:
            output["context"] = context
        if return_debug:
            debug.update(selected_ids=ids, selected_valid=valid)
            output["debug"] = debug
        if profile:
            output["timings_ms"] = timings
        return output
