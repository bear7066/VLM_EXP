import torch
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
a = torch.randn(2000, 2000, device=device)
b = torch.randn(2000, 2000, device=device)

torch.cuda.synchronize() if device == "cuda" else None
t1 = time.time()
c = a @ b
torch.cuda.synchronize() if device == "cuda" else None
t2 = time.time()

print("device =", device)
print("elapsed =", t2 - t1)
print("sum =", c.sum().item())