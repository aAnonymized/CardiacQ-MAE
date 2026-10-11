"""Independent, DIS-supervised coordinate queries. DIS is never a forward input."""
import torch
from torch import nn
from torch.nn import functional as F
from ..motion.geometry import phase_features


class CrossBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.norm = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2*dim)
        self.proj = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 4*dim), nn.GELU(), nn.Linear(4*dim, dim))

    def forward(self, q, context):
        b, n, d = q.shape
        h = self.heads
        query = self.q(self.norm(q)).reshape(b, n, h, d//h).transpose(1, 2)
        kv = self.kv(context).reshape(b, -1, 2, h, d//h).permute(2, 0, 3, 1, 4)
        if hasattr(F, "scaled_dot_product_attention"):
            out = F.scaled_dot_product_attention(query, kv[0], kv[1], dropout_p=0.)
        else:
            attention = (query @ kv[0].transpose(-1, -2))*(d//h)**-.5
            out = attention.softmax(-1) @ kv[1]
        q = q+self.proj(out.transpose(1, 2).reshape(b, n, d))
        return q+self.mlp(q)


class DirectQuery(nn.Module):
    def __init__(self, settings):
        super().__init__()
        self.options = settings["query"]
        self.size, self.frames = settings["data"]["input_size"], settings["data"]["num_frames"]
        d = self.options["dim"]
        self.context = nn.Linear(settings["model"]["embed_dim"], d)
        self.embedding = nn.Sequential(nn.Linear(26, d), nn.GELU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList([CrossBlock(d, self.options["heads"]) for _ in range(self.options["depth"])])
        self.output = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2))

    def forward(self, context, points, source_times, target_times):
        """points [B,Q,2] in pixels -> absolute predicted target xy in pixels."""
        p = points/(self.size-1)
        freq = (2.**torch.arange(4, device=p.device))*torch.pi
        angles = p[..., None]*freq
        features = torch.cat([p, angles.sin().flatten(-2), angles.cos().flatten(-2),
            phase_features(source_times, self.frames, self.options["phase_mode"]),
            phase_features(target_times, self.frames, self.options["phase_mode"])], -1)
        context = self.context(context)
        outputs = []
        for start in range(0, p.shape[1], self.options["chunk_size"]):
            q = self.embedding(features[:, start:start+self.options["chunk_size"]])
            for block in self.blocks:
                q = block(q, context)
            # Direct learned coordinates: not anchor + offset, and never DIS + residual.
            outputs.append(self.output(q)*(self.size-1))
        return torch.cat(outputs, 1)
