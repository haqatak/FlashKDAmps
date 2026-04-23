# FlashKDA

FlashKDA: Flash Kimi Delta Attention — high-performance KDA kernels built on CUTLASS

## Requirements
- SM90 and above
- CUDA 12.9 and above
- PyTorch 2.4 and above

## Installation
```bash
git clone https://github.com/MoonshotAI/FlashKDA.git flash-kda
cd flash-kda
git submodule update --init --recursive
pip install -v .
```

Once installed, FlashKDA can be used directly as a backend of `flash-linear-attention`. See [fla-org/flash-linear-attention#852](https://github.com/fla-org/flash-linear-attention/pull/852) for integration details.

## Apple Silicon / MPS Support

FlashKDA supports Apple Silicon (M-series chips) natively. When a CUDA device is not available, it automatically falls back to a pure PyTorch implementation optimized for MPS (`mps`).

To install on Apple Silicon without building the CUDA extensions, you simply run:

```bash
git clone https://github.com/MoonshotAI/FlashKDA.git flash-kda
cd flash-kda
pip install -v .
```

To run FlashKDA on MPS, ensure your tensors are placed on the `mps` device:

```python
import torch
import flash_kda

device = torch.device("mps")

# Create sample inputs
B, T, H, K = 1, 1024, 4, 128
V = K

q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device=device)
k = torch.randn(B, T, H, K, dtype=torch.bfloat16, device=device)
v = torch.randn(B, T, H, V, dtype=torch.bfloat16, device=device)
g = torch.randn(B, T, H, K, dtype=torch.bfloat16, device=device)
beta = torch.randn(B, T, H, dtype=torch.bfloat16, device=device)
scale = 1.0 / (K ** 0.5)
out = torch.zeros_like(v)

A_log = torch.randn(H, dtype=torch.float32, device=device)
dt_bias = torch.randn(H, K, dtype=torch.float32, device=device)
lower_bound = -5.0

# Run FlashKDA forward
flash_kda.fwd(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound)
```

## Performance

See [BENCHMARK_H20.md](BENCHMARK_H20.md).

## Tests

```bash
bash tests/test.sh
```

- `tests/test_fwd.py` — correctness tests (exact match against the torch reference; compared with `flash-linear-attention`)


## Kernel API

### `flash_kda.fwd`

```python
flash_kda.fwd(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound,
              initial_state=None, final_state=None, cu_seqlens=None)
```

**Parameters:**

| Parameter | Dtype | Shape | Description |
|---|---|---|---|
| `q` | bf16 | `[B, T, H, K]` | Query |
| `k` | bf16 | `[B, T, H, K]` | Key |
| `v` | bf16 | `[B, T, H, V]` | Value |
| `g` | bf16 | `[B, T, H, K]` | Gate before activation |
| `beta` | bf16 | `[B, T, H]` | Beta logits (pre-activation; sigmoid applied internally) |
| `scale` | float | scalar | scaling factor |
| `out` | bf16 | `[B, T, H, V]` | Output tensor |
| `A_log` | fp32 | `[H]` | Log-gate parameter |
| `dt_bias` | fp32 | `[H, K]` | Gate bias |
| `lower_bound` | float | scalar | Gate lower bound (range from -5.0 to 0) |
| `initial_state` | bf16/fp32/None | `[B, H, V, K]` or `[N, H, V, K]` | (optional) Initial recurrent state |
| `final_state` | bf16/fp32/None | `[B, H, V, K]` or `[N, H, V, K]` | (optional, output) Final recurrent state |
| `cu_seqlens` | int64 | `[N+1]` | (optional) Cumulative sequence lengths for variable-length batching |

- Currently requires `K = V = 128`.
- `initial_state` / `final_state` accept `None` (stateless), bf16, or fp32 tensors. When both are provided, their dtypes must match.
- When `cu_seqlens` is provided, `B` must be 1, `T` is the total length across all sequences, and `initial_state` / `final_state` have shape `[N, H, V, K]`.
- When `cu_seqlens` is `None`, each batch element is treated as an independent sequence, and the state shape is `[B, H, V, K]`.

## Development

To set up IntelliSense (clangd) for the CUDA/C++ sources, run:

```bash
bash setup_clangd.sh
```

This generates a `.clangd` file with the correct repository paths and installs the global clangd `config.yaml` to `~/.config/clangd/`.

## Citation

```bibtex
@misc{flashkda2026,
      title={FlashKDA: Flash Kimi Delta Attention},
      author={Yutian Chen, Zhiyuan Li, Yucheng Wang, Ming Wei},
      year={2026},
      publisher = {GitHub},
      howpublished = {\url{https://github.com/MoonshotAI/FlashKDA}},
}
```
