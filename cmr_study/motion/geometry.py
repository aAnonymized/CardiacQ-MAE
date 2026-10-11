import torch


def patch_anchors(size, patch, device="cpu"):
    axis = (torch.arange(size // patch, device=device).float() + .5) * patch - .5
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack([xx, yy], -1).reshape(-1, 2)


def local_points(anchors, patch):
    # The symmetric stencil stays inside the source patch.
    delta = patch / 4.
    offsets = anchors.new_tensor([[0, 0], [-delta, 0], [delta, 0], [0, -delta], [0, delta]])
    return anchors[:, None, :] + offsets[None]


def phase_features(times, frames, mode):
    t = times.float() / frames
    phase = (2 * torch.pi * t if mode == "uniform" else
             torch.where(t <= 2 / 3, 1.5 * torch.pi * t,
                         torch.pi + 3 * torch.pi * (t - 2 / 3)))
    # K_phi=2, matching the method text.
    return torch.stack([phase.sin(), phase.cos(), (2 * phase).sin(), (2 * phase).cos()], -1)


def jacobians(tracks, patch):
    """Finite-difference Jacobians for tracks [..., 5, 2]."""
    dx = (tracks[..., 2, :] - tracks[..., 1, :]) / (patch / 2.)
    dy = (tracks[..., 4, :] - tracks[..., 3, :]) / (patch / 2.)
    return torch.stack([dx, dy], -1)


def components(tracks, anchors, patch):
    """tracks [Q,T,5,2], K in pixels and R dimensionless."""
    k = (tracks[:, :, 0] - anchors[:, None]).norm(dim=-1).mean(1)
    identity = torch.eye(2, device=tracks.device, dtype=tracks.dtype)
    r = (jacobians(tracks, patch) - identity).square().sum((-1, -2)).sqrt().mean(1)
    return k, r


def normalize(values, options, component=None):
    """Absolute, experiment-wide scaling; never stretch each case separately."""
    if component == "k":
        scale = float(options["k_scale_px"])
    elif component == "r":
        scale = float(options["r_scale"])
    else:
        raise ValueError("normalize requires component='k' or component='r'")
    if scale <= 0:
        raise ValueError("TGWR normalization scales must be positive")
    return (values.float() / scale).clamp(0, 1)


def importance(k, r, options, component="full"):
    nk = normalize(k, options, "k")
    nr = normalize(r, options, "r")
    if component == "k":
        return nk.square()
    if component == "r":
        return nr.square()
    if component == "none":
        return torch.zeros_like(nk)
    if component != "full":
        raise ValueError("Unknown TGWR component: " + component)
    return .5 * (nk.square() + nr.square())
