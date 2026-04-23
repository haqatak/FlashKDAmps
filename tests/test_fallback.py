import torch
import pytest
from flash_kda.fallback import fp32_ex2_ftz

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
