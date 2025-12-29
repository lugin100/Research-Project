import os
import torch
from Plotting import plot_tensor_as_map
from Dataset import WeatherDataset, CoeffDataset
from Functions import (
		DEVICE,
		triangular_number,
		flatten_coeffs,
		unflatten_coeffs,
		sh_transform,
		inv_sh_transform)

model_str = "7vhmq3u0"
model_path = f"Results/{model_str}/"


ground_truth_path = "Results/ground-truth/"
data_path = "data/wind-speed_level-500_testset"

natural_ds = WeatherDataset("wind_speed", time_slice=slice("2011", "2012", None), level=500)
ground_truth = natural_ds.__getitem__(0)

T = triangular_number(121)
means = torch.load("data/wind-speed_level-500_testset_means.pt", weights_only=True)[None,:T]
stds = torch.load("data/wind-speed_level-500_testset_stds.pt", weights_only=True)[None,:T]

coeff_ds = CoeffDataset(data_path)
gt_coeff = coeff_ds.__getitem__(0)[:T]

gt_coeff = (gt_coeff * stds + means)
gt_reals = inv_sh_transform(unflatten_coeffs(gt_coeff)).squeeze().cpu()

if not os.path.exists(ground_truth_path):
	os.makedirs(ground_truth_path)

	# Plot single datapoint as example

	plot_tensor_as_map(ground_truth, save_name=ground_truth_path + "Ground-Truth", vmin=0, vmax=70)

	# Transform to coefficients
	single_datapoint_coeffs = flatten_coeffs(sh_transform(ground_truth[None,...]))

	L = triangular_number(61)
	# Plot training model input (coefficients zeroed after L)
	single_datapoint_coeffs[:,L:] = 0
	input_datapoint = inv_sh_transform(unflatten_coeffs(single_datapoint_coeffs))
	plot_tensor_as_map(input_datapoint.squeeze(), save_name=ground_truth_path + "Training-model-input", vmin=0, vmax=70)

	# Plot inference model input (only coefficients up to L)
	single_datapoint_coeffs = single_datapoint_coeffs[:,:L]
	input_datapoint = inv_sh_transform(unflatten_coeffs(single_datapoint_coeffs))
	plot_tensor_as_map(input_datapoint.squeeze(), save_name=ground_truth_path + "Inference-model-input", vmin=0, vmax=70)


# Load prediction
batch = torch.load(model_path + "Predictions/reals/batch_0.pt", weights_only=True)
first_sample = batch[0].cpu()

lower_halve = first_sample[61:,:]
plot_tensor_as_map(first_sample, save_name=model_path + "model-output")

diff = (first_sample - gt_reals)[61:,:]
plot_tensor_as_map(diff, save_name=model_path + "ground-truth-difference")
