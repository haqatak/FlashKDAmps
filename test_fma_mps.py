import torch
import platform
print(platform.system(), platform.machine())

device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
c = torch.tensor([1.0], dtype=torch.float32, device=device)
a = torch.tensor([2.0], dtype=torch.float32, device=device)
b = torch.tensor([3.0], dtype=torch.float32, device=device)

# test standard approach
try:
    res1 = (c.to(torch.float64) + a.to(torch.float64) * b.to(torch.float64)).to(torch.float32)
    print("res1", res1)
except Exception as e:
    print("res1 error:", type(e), e)

# test proposed approach
res2 = torch.addcmul(c, a, b)
print("res2", res2)

res3 = c + a * b
print("res3", res3)
