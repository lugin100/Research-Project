import torch
from torch.utils.data import DataLoader
from Dataset import CoeffDataset
from Functions import triangular_number, inv_sh_transform, unflatten_coeffs, DEVICE

N = 121
M = 60

T = triangular_number(N)
L = triangular_number(M)
ds = CoeffDataset("data/wind-speed_level-500_testset")

B = 10000
loader = DataLoader(ds, batch_size=B, num_workers=7, shuffle=False)

means = torch.load("data/wind-speed_level-500_testset_means.pt", weights_only=True)[None,:T].to(DEVICE)
stds = torch.load("data/wind-speed_level-500_testset_stds.pt", weights_only=True)[None,:T].to(DEVICE)

with torch.no_grad():
	for i, batch in enumerate(loader):
		batch[:,L:,:] = torch.randn_like(batch[:,L:,:])
		torch.save(torch.view_as_real(batch.cpu()), f"Results/random/Predictions/coeffs/batch_{i}.pt")
		batch = batch.to(DEVICE)
		batch = batch * stds + means
		result = inv_sh_transform(unflatten_coeffs(batch))
		torch.save(result.cpu(), f"Results/random/Predictions/reals/batch_{i}.pt")
