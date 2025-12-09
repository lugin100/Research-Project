import torch
import torch.nn.Functional as F
from torch.utils.data import DataLoader
from Dataset import CoeffDataset
from Functions import inv_sh_transform, unflatten_coeffs, DEVICE

N = 120
M = 60

T = N*(N+1)/2
L = M*(M+1)/2

ds = CoeffDataset("data/wind-speed_level-500_testset", index_limit=L)

B = 4
loader = DataLoader(ds, batch_size=B, num_workers=7, shuffle=False)

with torch.no_grad():
	for i, batch in enumerate(loader):
		batch = batch.to(DEVICE)
		input = inv_sh_transform(unflatten_coeffs(batch)).unsqueeze(1) # (B,1,M,M)
		result = F.interpolate(input, size=(N,N), mode="bilinear")
		result = result.cpu().squeeze(1) # (B,N,N)
		torch.save(result, f"Results/interpolation/batch_{i}")
