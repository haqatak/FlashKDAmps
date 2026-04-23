import torch
import pytest
from flash_kda.fallback import l2_normalize_kernel_match

@pytest.mark.parametrize("shape", [
    (128,),                 # Minimal shape
    (10, 128),              # 2D shape
    (4, 5, 128),            # 3D shape
    (2, 3, 4, 128),         # 4D shape (typical for B, T, H, D)
])
def test_l2_normalize_kernel_match(shape):
    """Test L2 normalization matches theoretical expected values."""
    torch.manual_seed(42)

    # Generate random input in bfloat16 as expected by the function
    x = torch.randn(shape, dtype=torch.bfloat16)

    # Run the function under test
    actual = l2_normalize_kernel_match(x)

    # Calculate the expected value
    x_f32 = x.float()
    expected_norm = torch.sqrt((x_f32 ** 2).sum(dim=-1, keepdim=True) + 1e-6)
    expected = (x_f32 / expected_norm).to(torch.bfloat16)

    # Verify the output matches the expected result
    # We use a looser tolerance (1e-2) to account for different calculation paths
    # (FMA tree reduction vs native ATen reduction) causing slight differences
    # in intermediate float32 values, which are magnified when cast back to bfloat16.
    assert torch.allclose(actual.float(), expected.float(), atol=1e-2, rtol=1e-2), \
        "Output does not match expected theoretical L2 normalization"

    # Verify the output type
    assert actual.dtype == torch.bfloat16, "Output dtype should be bfloat16"

    # Verify the output shape
    assert actual.shape == shape, "Output shape should match input shape"

def test_l2_normalize_kernel_match_invalid_shape():
    """Test that the function raises an error or behaves as expected for invalid shapes."""
    # The docstring says D must be 128, which is enforced by the reshaping inside the function:
    # groups = x_f32.reshape(*x_f32.shape[:-1], 16, 8)
    # If D is not 128, this reshape will fail.

    x = torch.randn((10, 64), dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="shape"):
        l2_normalize_kernel_match(x)
