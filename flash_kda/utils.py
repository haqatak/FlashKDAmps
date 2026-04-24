import torch

# ============================================================
# Numeric helpers
# ============================================================

LOG2E = 1.4426950408889634


def fp32_ex2_ftz(x):
    if x.dtype == torch.float16:
        x = x.to(torch.float32)
    ret = torch.special.exp2(x)
    ret = torch.where(ret.abs() < torch.finfo(torch.float32).tiny, torch.zeros_like(ret), ret)
    return ret


def fp32_fma(c, a, b):
    if c.dtype != torch.float32:
        raise TypeError(f"Expected c.dtype to be torch.float32, got {c.dtype}")
    if a.dtype != torch.float32:
        raise TypeError(f"Expected a.dtype to be torch.float32, got {a.dtype}")
    if b.dtype != torch.float32:
        raise TypeError(f"Expected b.dtype to be torch.float32, got {b.dtype}")

    if c.device.type == 'mps':
        return torch.addcmul(c, a, b)

    return (c.to(torch.float64) + a.to(torch.float64) * b.to(torch.float64)).to(torch.float32)


def l2_normalize_kernel_match(x):
    """L2 normalize matching kernel's warp-shuffle tree reduction with FMA.
    x: [..., D] bf16, D must be 128.
    """
    x_f32 = x.float()
    groups = x_f32.reshape(*x_f32.shape[:-1], 16, 8)

    partials = torch.zeros(*x_f32.shape[:-1], 16, dtype=torch.float32, device=x.device)
    for i in range(8):
        partials = fp32_fma(partials, groups[..., i], groups[..., i])

    for offset in [8, 4, 2, 1]:
        indices = torch.arange(16, device=x.device) ^ offset
        partials = partials + partials[..., indices]

    inv_norm = torch.rsqrt(partials[..., 0:1] + 1e-6)
    return (x_f32 * inv_norm).to(x.dtype)
