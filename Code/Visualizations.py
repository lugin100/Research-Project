# Visualizations on first sample

import numpy as np
from matplotlib import pyplot as plt
import torch
from Dataset import CoeffDataset
from Functions import (
		DEVICE,
		triangular_number,
		flatten_coeffs,
		unflatten_coeffs,
		sh_transform,
		inv_sh_transform)

data_path = "data/wind-speed_level-500_testset"

means = torch.load(data_path + "_means.pt", weights_only=True)
stds = torch.load(data_path + "_stds.pt", weights_only=True)

coeff_ds = CoeffDataset(data_path)
gt_coeff = coeff_ds.__getitem__(0)

gt_coeff = (gt_coeff * stds + means)
gt_reals = inv_sh_transform(unflatten_coeffs(gt_coeff[None])).squeeze().cpu()

L = triangular_number(61)

vmin = -30
vmax = 40

fig, subplots = plt.subplots(nrows = 4, ncols = 2, sharex="all", sharey="all")


subplots[0,0].imshow(gt_reals, cmap = "turbo", vmin=vmin, vmax=vmax)

gt_coeff_cutoff = gt_coeff[:L]
gt_reals_input = inv_sh_transform(unflatten_coeffs(gt_coeff_cutoff[None])).squeeze()

subplots[0,1].imshow(gt_reals_input, cmap = "turbo", vmin=vmin, vmax=vmax)

models = ["interpolation", "New-model", "New-model-deep"]
for i, model in enumerate(models):
    batch = torch.load(f"Results/{model}/Predictions/reals/batch_0.pt", weights_only=True, map_location="cpu")
    first_sample = batch[0].cpu()
    subplots[i+1,0].imshow(first_sample, cmap = "turbo", vmin=vmin, vmax=vmax)

    difference = first_sample - gt_reals
    im = subplots[i+1,1].imshow(difference, cmap = "turbo", vmin=vmin, vmax=vmax)

fig.supxlabel("Longitude")
fig.supylabel("Latitude")
fig.colorbar(im, ax=subplots.ravel().tolist())
fig.savefig("Figures/Sample-Visualization.pdf",  bbox_inches = "tight")
