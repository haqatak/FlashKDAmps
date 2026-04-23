import torch
import pytest
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
