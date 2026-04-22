import torch
import pytest
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
