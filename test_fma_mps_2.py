import torch

c = torch.tensor([1.0], dtype=torch.float32)
a = torch.tensor([2.0], dtype=torch.float32)
b = torch.tensor([3.0], dtype=torch.float32)

# test standard approach, simulating mps
class MockTensor:
    def __init__(self, tensor):
        self.tensor = tensor
        self.dtype = tensor.dtype
        self.device = type('Device', (), {'type': 'mps'})()

    def to(self, dtype):
        if dtype == torch.float64:
             raise TypeError("Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.")
        return MockTensor(self.tensor.to(dtype))

print("OK")
