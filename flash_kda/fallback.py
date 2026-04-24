import torch

from .utils import LOG2E, fp32_ex2_ftz, fp32_fma, l2_normalize_kernel_match

# ============================================================
# Fallback implementations without CUDA custom kernels
# ============================================================

def sigmoid_tanh_fp32(input):
    return torch.tanh(input * 0.5) * 0.5 + 0.5

def matmul_fp16acc(a, b):
    return torch.matmul(a.to(torch.float32), b.to(torch.float32)).to(torch.float16)

def fwd_fallback(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound, initial_state=None, final_state=None, cu_seqlens=None):
    """Torch reference fallback, supports both fixed-length and variable-length sequences.

    Input: [B, T, H, D] (4D). B must be 1 when cu_seqlens is provided.
    """
    if q.dim() != 4:
        raise ValueError(f"Expected 4D input [B, T, H, D], got {q.dim()}D")
    B = q.shape[0]
    if cu_seqlens is not None:
        if B != 1:
            raise ValueError(f"B must be 1 when cu_seqlens is provided, got B={B}")

    q = q.reshape(-1, *q.shape[2:])
    k = k.reshape(-1, *k.shape[2:])
    v = v.reshape(-1, *v.shape[2:])
    g = g.reshape(-1, *g.shape[2:])
    beta = beta.reshape(-1, *beta.shape[2:])
    out = out.reshape(-1, *out.shape[2:])

    if B > 1:
        T_seq = q.shape[0] // B
        cu_seqlens = torch.arange(0, B * T_seq + 1, T_seq, dtype=torch.long, device=q.device)

    _, H, D = q.shape
    CHUNK = 16
    device = q.device
    scale_bf16 = torch.tensor(scale, dtype=torch.bfloat16, device=device)

    q = l2_normalize_kernel_match(q)
    k = l2_normalize_kernel_match(k)

    if A_log is not None:
        if dt_bias is None:
            raise ValueError("dt_bias must be provided when A_log is provided")
        if A_log.dtype != torch.float32:
            raise TypeError(f"Expected A_log.dtype to be torch.float32, got {A_log.dtype}")
        if g.dtype != torch.bfloat16:
            raise TypeError(f"Expected g.dtype to be torch.bfloat16, got {g.dtype}")
        if dt_bias.dtype != torch.float32:
            raise TypeError(f"Expected dt_bias.dtype to be torch.float32, got {dt_bias.dtype}")
        g = g.to(torch.float32) + dt_bias.unsqueeze(0)
        a_log_exp = fp32_ex2_ftz(A_log * LOG2E).unsqueeze(0).unsqueeze(-1)
        scale = lower_bound * LOG2E
        g = scale * sigmoid_tanh_fp32(a_log_exp * g)

    state_fp32 = (initial_state is not None and initial_state.dtype == torch.float32) or \
                 (final_state is not None and final_state.dtype == torch.float32)

    if cu_seqlens is None:
        T = q.shape[0]
        cu_seqlens = torch.tensor([0, T], dtype=torch.long, device=device)

    N = len(cu_seqlens) - 1

    if initial_state is not None:
        work_state = initial_state.to(torch.bfloat16).clone()
    else:
        work_state = torch.zeros(N, H, D, D, dtype=torch.bfloat16, device=device)

    for seq_idx in range(N):
        bos = cu_seqlens[seq_idx].item()
        eos = cu_seqlens[seq_idx + 1].item()
        seq_len = eos - bos
        n_chunks = (seq_len + CHUNK - 1) // CHUNK

        for chunk_idx in range(n_chunks):
            t0 = bos + chunk_idx * CHUNK
            actual_len = min(CHUNK, eos - t0)

            for h in range(H):
                g_chunk = torch.zeros(CHUNK, D, dtype=g.dtype, device=device)
                q_chunk = torch.zeros(CHUNK, D, dtype=q.dtype, device=device)
                k_chunk = torch.zeros(CHUNK, D, dtype=k.dtype, device=device)
                v_chunk = torch.zeros(CHUNK, D, dtype=v.dtype, device=device)
                beta_chunk = torch.zeros(CHUNK, dtype=beta.dtype, device=device)

                g_chunk[:actual_len] = g[t0:t0 + actual_len, h, :]
                q_chunk[:actual_len] = q[t0:t0 + actual_len, h, :]
                k_chunk[:actual_len] = k[t0:t0 + actual_len, h, :]
                v_chunk[:actual_len] = v[t0:t0 + actual_len, h, :]
                beta_chunk[:actual_len] = beta[t0:t0 + actual_len, h]

                g_cumsum = g_chunk.cumsum(dim=0)
                g_total = g_cumsum[-1:]
                k_decayed = k_chunk * fp32_ex2_ftz(g_cumsum).to(torch.bfloat16)
                q_decayed = q_chunk * fp32_ex2_ftz(g_cumsum).to(torch.bfloat16) * scale_bf16
                neg_g_cumsum_bf16 = fp32_ex2_ftz(-g_cumsum).to(torch.bfloat16)
                k_inv = k_chunk * neg_g_cumsum_bf16
                g_total_exp_bf16 = fp32_ex2_ftz(g_total).to(torch.bfloat16)
                k_restored = k_inv * g_total_exp_bf16
                L = torch.matmul(k_decayed.to(torch.float32), k_inv.t().to(torch.float32)).to(torch.float16)
                Mqk = torch.matmul(q_decayed, k_inv.t())

                # Fuse sigmoid via tanh.approx: beta is bf16 logits
                beta_activated = sigmoid_tanh_fp32(beta_chunk.to(torch.float32))
                beta_val_bf16 = beta_activated.to(torch.bfloat16).unsqueeze(-1)
                beta_val_fp16 = beta_activated.to(torch.float16).unsqueeze(-1)
                L = torch.tril(L, diagonal=-1) * beta_val_fp16
                Mqk = torch.tril(Mqk)

                INV = torch.eye(CHUNK, dtype=torch.float16, device=device) - L
                L2 = matmul_fp16acc(L, L)
                INV = INV + matmul_fp16acc(INV, L2)
                L4 = matmul_fp16acc(L2, L2)
                INV = INV + matmul_fp16acc(INV, L4)
                L8 = matmul_fp16acc(L4, L4)
                INV = INV + matmul_fp16acc(INV, L8)

                INV = INV.to(torch.bfloat16)

                state_slice = work_state[seq_idx, h]
                v_chunk = v_chunk - torch.matmul(k_decayed, state_slice.t())
                v_chunk = v_chunk * beta_val_bf16

                U = torch.matmul(INV, v_chunk)
                _out = torch.matmul(q_decayed, state_slice.t())
                _out = _out + torch.matmul(Mqk, U)

                delta_s = torch.matmul(k_restored.t().to(torch.float32), U.to(torch.float32))

                g_total_exp = fp32_ex2_ftz(g_total)
                g_total_exp = g_total_exp.squeeze(0).unsqueeze(-1)
                work_state[seq_idx, h] = fp32_fma(delta_s, state_slice.to(torch.float32).t(), g_total_exp).to(torch.bfloat16).t()

                out[t0:t0 + actual_len, h] = _out[:actual_len]

    if final_state is not None:
        if state_fp32:
            final_state.copy_(work_state.to(torch.float32))
        else:
            final_state.copy_(work_state)
