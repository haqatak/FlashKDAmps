import torch
import pytest
from flash_kda.fallback import fp32_ex2_ftz
from unittest.mock import patch, MagicMock
import flash_kda

def test_fp32_ex2_ftz():
    # Test typical float32 values
    x_f32 = torch.tensor([0.0, 1.0, -1.0, 2.0, -2.0], dtype=torch.float32)
    expected_f32 = torch.special.exp2(x_f32)
    res_f32 = fp32_ex2_ftz(x_f32)
    assert torch.allclose(res_f32, expected_f32)
    assert res_f32.dtype == torch.float32

    # Test float16 input is promoted to float32
    x_f16 = torch.tensor([0.0, 1.0, -1.0], dtype=torch.float16)
    expected_f16 = torch.special.exp2(x_f16.to(torch.float32))
    res_f16 = fp32_ex2_ftz(x_f16)
    assert torch.allclose(res_f16, expected_f16)
    assert res_f16.dtype == torch.float32

    # Test flush-to-zero logic
    # torch.finfo(torch.float32).tiny is ~1.17549435e-38, which is 2^-126
    # So exp2(x) < 2^-126 will flush to zero.

    # Values that should not flush to zero
    x_no_flush = torch.tensor([-125.0, -125.5], dtype=torch.float32)
    res_no_flush = fp32_ex2_ftz(x_no_flush)
    assert torch.all(res_no_flush > 0.0)
    assert torch.allclose(res_no_flush, torch.special.exp2(x_no_flush))

    # Values that should flush to zero
    # -126.0 might result exactly in 2^-126 depending on implementation,
    # but strictly speaking `ret.abs() < tiny` will evaluate to false if ret == tiny.
    # Actually `torch.finfo(torch.float32).tiny` is exactly 2^-126.
    # If x = -126.0, exp2(x) = 2^-126, so `abs() < tiny` is False.
    # Wait, let's test -127.0 which is definitely < 2^-126
    x_flush = torch.tensor([-127.0, -150.0], dtype=torch.float32)
    res_flush = fp32_ex2_ftz(x_flush)
    assert torch.all(res_flush == 0.0)


def test_fwd_fallback_dispatch():
    # Setup dummy tensors (on CPU, so q.device.type == "cpu")
    B, T, H, K, V = 1, 2, 4, 128, 128
    q = torch.randn(B, T, H, K, dtype=torch.bfloat16)
    k = torch.randn(B, T, H, K, dtype=torch.bfloat16)
    v = torch.randn(B, T, H, V, dtype=torch.bfloat16)
    g = torch.randn(B, T, H, K, dtype=torch.bfloat16)
    beta = torch.randn(B, T, H, dtype=torch.bfloat16)
    scale = 0.1
    out = torch.zeros_like(v)
    A_log = torch.randn(H, dtype=torch.float32)
    dt_bias = torch.randn(H, K, dtype=torch.float32)
    lower_bound = -5.0
    initial_state = torch.randn(B, H, V, K, dtype=torch.bfloat16)
    final_state = torch.randn(B, H, V, K, dtype=torch.bfloat16)
    cu_seqlens = torch.tensor([0, T], dtype=torch.int64)

    # 1. Test when _HAS_EXT is True, but device is CPU -> should fallback
    with patch("flash_kda._HAS_EXT", True), \
         patch("flash_kda.fwd_fallback") as mock_fallback:

        flash_kda.fwd(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound,
                      initial_state=initial_state, final_state=final_state, cu_seqlens=cu_seqlens)

        mock_fallback.assert_called_once()
        # Verify it was called with the exact arguments
        args, kwargs = mock_fallback.call_args
        assert args[0] is q
        assert args[1] is k
        assert args[2] is v
        assert args[3] is g
        assert args[4] is beta
        assert args[5] == scale
        assert args[6] is out
        assert args[7] is A_log
        assert args[8] is dt_bias
        assert args[9] == lower_bound
        assert kwargs["initial_state"] is initial_state
        assert kwargs["final_state"] is final_state
        assert kwargs["cu_seqlens"] is cu_seqlens

    # 2. Test when _HAS_EXT is False -> should fallback regardless of device
    with patch("flash_kda._HAS_EXT", False), \
         patch("flash_kda.fwd_fallback") as mock_fallback:

        flash_kda.fwd(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound)

        mock_fallback.assert_called_once()
        args, kwargs = mock_fallback.call_args
        assert args[0] is q
        assert kwargs["initial_state"] is None
        assert kwargs["final_state"] is None
        assert kwargs["cu_seqlens"] is None
