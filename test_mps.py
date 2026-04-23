import torch
from flash_kda.utils import fp32_fma

def my_fma(c, a, b):
    if c.device.type == 'mps':
        return torch.addcmul(c, a, b)
    return (c.to(torch.float64) + a.to(torch.float64) * b.to(torch.float64)).to(torch.float32)

c = torch.tensor([1.0], dtype=torch.float32)
a = torch.tensor([2.0], dtype=torch.float32)
b = torch.tensor([3.0], dtype=torch.float32)
print(fp32_fma(c, a, b))
print(my_fma(c, a, b))
