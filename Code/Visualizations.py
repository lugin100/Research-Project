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

model_str = "interpolation"
model_path = f"Results/{model_str}/"


ground_truth_path = "Results/ground-truth/"
data_path = "data/wind-speed_level-500_testset"

natural_ds = WeatherDataset("wind_speed", time_slice=slice("2011", "2012", None), level=500)
ground_truth = natural_ds.__getitem__(0)

T = 7381
means = torch.load("data/wind-speed_level-500_testset_means.pt", weights_only=True)[None,:T]
stds = torch.load("data/wind-speed_level-500_testset_stds.pt", weights_only=True)[None,:T]

coeff_ds = CoeffDataset(data_path)
gt_coeff = coeff_ds.__getitem__(0)

gt_coeff = (gt_coeff * stds + means)[:,:T]
gt_reals = inv_sh_transform(unflatten_coeffs(gt_coeff)).squeeze().cpu()

if not os.path.exists(ground_truth_path):
	os.makedirs(ground_truth_path)

	# Plot single datapoint as example
	
	plot_tensor_as_map(ground_truth, save_name=ground_truth_path + "Ground-Truth")

	# Transform to coefficients
	single_datapoint_coeffs = flatten_coeffs(sh_transform(ground_truth[None,...]))

	L = triangular_number(60)
	# Plot training model input (coefficients zeroed after L)
	single_datapoint_coeffs[:,L:] = 0
	input_datapoint = inv_sh_transform(unflatten_coeffs(single_datapoint_coeffs))
	plot_tensor_as_map(input_datapoint.squeeze(), save_name=ground_truth_path + "Training-model-input")

	# Plot inference model input (only coefficients up to L)
	single_datapoint_coeffs = single_datapoint_coeffs[:,:L]
	input_datapoint = inv_sh_transform(unflatten_coeffs(single_datapoint_coeffs))
	plot_tensor_as_map(input_datapoint.squeeze(), save_name=ground_truth_path + "Inference-model-input")


# Load prediction
batch = torch.load(model_path + "Predictions/reals/batch_0.pt", weights_only=True)
first_sample = batch[0].cpu()

plot_tensor_as_map(first_sample, save_name=model_path + "model-output")

diff = first_sample - gt_reals
plot_tensor_as_map(diff, save_name=model_path + "ground-truth-difference")
