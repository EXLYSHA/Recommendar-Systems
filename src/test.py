import torch, time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
A = torch.randn(1000,1000, device=device)
B = torch.randn(1000,1000, device=device)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(1000):
    C = A @ B
torch.cuda.synchronize()
print("GPU 矩阵乘 1000 次耗时：", time.time()-t0)