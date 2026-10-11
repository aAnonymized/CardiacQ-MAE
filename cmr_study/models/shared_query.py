"""One pointwise cross-attention decoder; pixel and coordinate-query heads."""
import torch
from torch import nn
from torch.nn import functional as F
from ..motion.geometry import phase_features


class SharedCrossBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.norm = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.proj = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 4 * dim),
                                 nn.GELU(), nn.Linear(4 * dim, dim))

    def project_context(self, context):
        b, length, d = context.shape
        return self.kv(context).reshape(b, length, 2, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)

    def forward(self, tokens, kv):
        b, n, d = tokens.shape
        query = self.q(self.norm(tokens)).reshape(b, n, self.heads, d // self.heads).transpose(1, 2)
        if hasattr(F, "scaled_dot_product_attention"):
            result = F.scaled_dot_product_attention(query, kv[0], kv[1], dropout_p=0.)
        else:
            attention = (query @ kv[0].transpose(-1, -2)) * (d // self.heads) ** -.5
            result = attention.softmax(-1) @ kv[1]
        tokens = tokens + self.proj(result.transpose(1, 2).reshape(b, n, d))
        return tokens + self.mlp(tokens)


class SharedQueryDecoder(nn.Module):
    """Task 0 reconstructs a tubelet; task 1 predicts absolute target xy.

    Prepared K/V lives only in the calling forward, retaining its autograd graph.
    No image crops, DIS coordinates or query-to-query attention are used.
    """
    def __init__(self, settings):
        super().__init__()
        q, options = settings["query"], settings.get("shared_decoder", {})
        d = int(options.get("dim", q["dim"]))
        heads = int(options.get("heads", q["heads"]))
        depth = int(options.get("depth", q["depth"]))
        self.chunk_size = int(q["chunk_size"])
        self.size = settings["data"]["input_size"]
        self.frames = settings["data"]["num_frames"]
        self.phase_mode = q["phase_mode"]
        self.max_displacement = float(q.get("max_displacement_px", 3 * settings["model"]["patch_size"]))
        self.pixel_dim = settings["model"]["tubelet_size"] * settings["model"]["patch_size"] ** 2
        if min(d, heads, depth, self.chunk_size, self.max_displacement) <= 0 or d % heads:
            raise ValueError("Positive shared decoder dimensions and dim divisible by heads required")
        if self.phase_mode not in ("uniform", "fixed_two_thirds"):
            raise ValueError("Unknown CCPE mode: " + self.phase_mode)
        self.context = nn.Linear(settings["model"]["embed_dim"], d)
        self.embedding = nn.Sequential(nn.Linear(26, d), nn.GELU(), nn.Linear(d, d))
        self.task_embedding = nn.Embedding(2, d)
        self.blocks = nn.ModuleList([SharedCrossBlock(d, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(d)
        self.pixel_head = nn.Linear(d, self.pixel_dim)
        self.coordinate_head = nn.Linear(d, 2)
        # The motion head still returns an absolute target coordinate.  This
        # source-relative parameterization only supplies a stable identity
        # initialization; it never consumes or adds a DIS prediction.
        nn.init.zeros_(self.coordinate_head.weight)
        nn.init.zeros_(self.coordinate_head.bias)

    def prepare(self, context):
        projected = self.context(context)
        return tuple(block.project_context(projected) for block in self.blocks)

    @staticmethod
    def select_context(prepared, indices):
        return tuple(kv.index_select(1, indices) for kv in prepared)

    def decode(self, prepared, points, source_times, target_times, task):
        if task not in (0, 1) or points.ndim != 3 or points.shape[-1] != 2:
            raise ValueError("Expected task 0/1 and query points [B,Q,2]")
        if source_times.shape != points.shape[:2] or target_times.shape != points.shape[:2]:
            raise ValueError("Query time arrays must match [B,Q]")
        if points.shape[1] == 0:
            return points.new_empty(points.shape[0], 0, self.pixel_dim if task == 0 else 2)
        outputs = []
        for start in range(0, points.shape[1], self.chunk_size):
            end = start + self.chunk_size
            p = points[:, start:end] / (self.size - 1)
            freq = (2. ** torch.arange(4, device=p.device)) * torch.pi
            angles = p[..., None] * freq
            features = torch.cat([p, angles.sin().flatten(-2), angles.cos().flatten(-2),
                phase_features(source_times[:, start:end], self.frames, self.phase_mode),
                phase_features(target_times[:, start:end], self.frames, self.phase_mode)], -1)
            tokens = self.embedding(features) + self.task_embedding.weight[task]
            for block, kv in zip(self.blocks, prepared):
                tokens = block(tokens, kv)
            tokens = self.norm(tokens)
            if task == 0:
                output = self.pixel_head(tokens)
            else:
                raw_delta = torch.tanh(self.coordinate_head(tokens))
                # Respect both the configured cardiac-motion range and the
                # available image extent without a zero-gradient clamp.
                room = torch.where(raw_delta >= 0, self.size - 1 - points[:, start:end],
                                   points[:, start:end]).clamp_max(self.max_displacement)
                output = points[:, start:end] + raw_delta * room
                same_phase = (source_times[:, start:end] == target_times[:, start:end]).unsqueeze(-1)
                output = torch.where(same_phase, points[:, start:end], output)
            outputs.append(output)
        return torch.cat(outputs, 1)

    def forward(self, context, points, source_times, target_times):
        # Compatibility with the original trajectory-only public interface.
        return self.decode(self.prepare(context), points, source_times, target_times, task=1)
