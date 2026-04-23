import torch
import pytest
from flash_kda.fallback import fp32_ex2_ftz
from unittest.mock import patch, MagicMock
import flash_kda
from flash_kda.fallback import sigmoid_tanh_fp32

def test_sigmoid_tanh_fp32():
    # Test random inputs
    torch.manual_seed(42)
    x = torch.randn(10, 10, dtype=torch.float32)
    res = sigmoid_tanh_fp32(x)

    # Check against the exact formula used
    expected_formula = torch.tanh(x * 0.5) * 0.5 + 0.5
    assert torch.allclose(res, expected_formula)

    # Check against actual sigmoid (should be a good approximation)
    expected_sigmoid = torch.sigmoid(x)
    # The approximation might have a small error compared to true sigmoid
    # According to our sandbox test, it's very close!
    assert torch.allclose(res, expected_sigmoid, atol=1e-2)

    # Test extreme values (should clamp to 0 and 1)
    x_ext = torch.tensor([-100.0, 100.0], dtype=torch.float32)
    res_ext = sigmoid_tanh_fp32(x_ext)
    assert torch.allclose(res_ext, torch.tensor([0.0, 1.0], dtype=torch.float32))
from flash_kda.fallback import fp32_ex2_ftz, fp32_fma

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
def test_fp32_fma():
    # 1. Test basic correct computation and output type
    c = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    a = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float32)
    b = torch.tensor([7.0, 8.0, 9.0], dtype=torch.float32)

    res = fp32_fma(c, a, b)
    expected = torch.tensor([29.0, 42.0, 57.0], dtype=torch.float32)

    torch.testing.assert_close(res, expected)
    assert res.dtype == torch.float32

    # 2. Test assertion failures for non-float32 inputs
    with pytest.raises(AssertionError):
        fp32_fma(c.to(torch.float64), a, b)
    with pytest.raises(AssertionError):
        fp32_fma(c, a.to(torch.float16), b)
    with pytest.raises(AssertionError):
        fp32_fma(c, a, b.to(torch.bfloat16))

    # 3. Test precision benefits of intermediate float64
    # We choose values that fit exactly in float32, but their product requires more than 24 bits.
    # a = 4097.0 (2^12 + 1), b = 4097.0 (2^12 + 1)
    # a * b = 16785409.0, which needs 25 bits.
    # In float32, a * b is rounded to 16785408.0.
    # So if we subtract 16785408.0 (c = -16785408.0), a native float32 `c + a * b` gives 0.0.
    # But doing it in float64 intermediate gives `16785409.0 - 16785408.0 = 1.0`.
    a_prec = torch.tensor([4097.0], dtype=torch.float32)
    b_prec = torch.tensor([4097.0], dtype=torch.float32)
    c_prec = torch.tensor([-16785408.0], dtype=torch.float32)

    res_prec = fp32_fma(c_prec, a_prec, b_prec)
    res_naive = c_prec + a_prec * b_prec

    # In pure float32, this evaluates to 0.0
    assert res_naive.item() == 0.0

    # With intermediate float64, it correctly retains the lost precision and returns 1.0
    assert res_prec.item() == 1.0
