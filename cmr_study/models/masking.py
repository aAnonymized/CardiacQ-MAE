import torch


def make_mask(batch, temporal, spatial, ratio, device, strategy="random", seed=None, motion=None, rho=.5):
    generator = None if seed is None else torch.Generator(device=device).manual_seed(int(seed))
    length = temporal*spatial
    noise = torch.rand(batch, length if strategy == "random" else spatial, device=device, generator=generator)
    if strategy == "random":
        keep = max(1, min(length-1, int(length*(1-ratio))))
        order = noise.argsort(-1)
        mask = torch.ones(batch, length, device=device)
        mask.scatter_(1, order[:, :keep], 0)
        return mask
    keep = max(1, min(spatial-1, int(spatial*(1-ratio))))
    if strategy == "motion":
        if motion is None:
            raise ValueError("TrackMAE motion-aware mask needs CoTracker tracks")
        ranked = motion.argsort(-1, descending=True)
        nh = (spatial+1)//2
        hi, lo = ranked[:, :nh], ranked[:, nh:]
        kh = min(nh, int(rho*keep))
        kl = min(lo.shape[1], keep-kh)
        kh = keep-kl
        high_noise = torch.gather(noise, 1, hi).argsort(-1)[:, :kh]
        low_noise = torch.gather(noise, 1, lo).argsort(-1)[:, :kl]
        chosen = torch.cat([hi.gather(1, high_noise), lo.gather(1, low_noise)], -1)
    else:
        chosen = noise.argsort(-1)[:, :keep]
    base = torch.ones(batch, spatial, device=device)
    base.scatter_(1, chosen, 0)
    return base[:, None].expand(-1, temporal, -1).reshape(batch, length)
