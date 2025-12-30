import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from Dataset import CoeffDataset
from Functions import triangular_number, sh_transform, flatten_coeffs, inv_sh_transform, unflatten_coeffs, DEVICE

N = 121
M = 61

T = triangular_number(N)
L = triangular_number(M)

means = torch.load("data/wind-speed_level-500_trainset_means.pt", weights_only=True).to(DEVICE)
stds = torch.load("data/wind-speed_level-500_trainset_stds.pt", weights_only=True).to(DEVICE)


ds = CoeffDataset("data/wind-speed_level-500_testset", index_limit=L)

B = 256
loader = DataLoader(ds, batch_size=B, num_workers=7, shuffle=False)

with torch.no_grad():
	for i, batch in enumerate(loader):
		batch = batch.to(DEVICE)
		batch = batch * stds[None,:L] + means[None,:L]
		input = inv_sh_transform(unflatten_coeffs(batch)).unsqueeze(1)
		result = F.interpolate(input, size=(N, 2*(N-1)), mode="bilinear")
		result = result.squeeze(1)
		torch.save(result.cpu(), f"Results/interpolation/Predictions/reals/batch_{i}.pt")

		coeffs = flatten_coeffs(sh_transform(result))
		coeffs = (coeffs - means) / stds
		coeffs = torch.view_as_real(coeffs)
		torch.save(coeffs.cpu(), f"Results/interpolation/Predictions/coeffs/batch_{i}.pt")
