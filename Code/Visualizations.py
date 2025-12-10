import os
import torch
from Plotting import plot_tensor_as_map
from Dataset import WeatherDataset
from Functions import (
		DEVICE,
		triangular_number,
		flatten_coeffs,
		unflatten_coeffs,
		sh_transform,
		inv_sh_transform)

from Transformer import LightningModel

model_str = "mitogrw5"
model_path = f"Results/{model_str}/"

ground_truth_path = "Results/ground-truth"
if not os.path.exists(ground_truth_path):
    os.makedirs(ground_truth_path)
	
	natural_ds = WeatherDataset("wind_speed", time_slice=slice("2011", "2012", None), level=500)

	# Plot single datapoint as example
	single_datapoint = natural_ds.__getitem__(0)
	plot_tensor_as_map(single_datapoint, save_name=ground_truth_path + "Ground-Truth")

	# Transform to coefficients
	single_datapoint_coeffs = flatten_coeffs(sh_transform(single_datapoint[None,...]))

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
first_sample = batch[0]

plot_tensor_as_map(first_sample, save_name=model_path + "Inference-model-output")
