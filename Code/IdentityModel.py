import torch
from torch.utils.data import DataLoader
from Dataset import CoeffDataset
from Functions import inv_sh_transform, unflatten_coeffs, DEVICE

N = 121

T = int(N*(N+1)/2)

ds = CoeffDataset("data/wind-speed_level-500_testset")

B = 1000
loader = DataLoader(ds, batch_size=B, num_workers=7, shuffle=False)

with torch.no_grad():
	for i, batch in enumerate(loader):
		batch = batch.to(DEVICE)
		result = inv_sh_transform(unflatten_coeffs(batch))
		torch.save(result.cpu(), f"Results/identity/Predictions/reals/batch_{i}.pt")
