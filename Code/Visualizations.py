import torch
from Plotting import plot_tensor_as_map
from Dataset import WeatherDataset
from Functions import (
		DEVICE,
		flatten_coeffs,
		unflatten_coeffs,
		sh_transform,
		inv_sh_transform)

from Transformer import LightningModel

model_str = "mitogrw5"
DIR = f"Results/{model_str}/"

natural_ds = WeatherDataset("wind_speed", time_slice=slice("2011", "2012", None), level=500)

# Works for full size model
L = int(60*61/2)

# Plot single datapoint as example
single_datapoint = natural_ds.__getitem__(0)
plot_tensor_as_map(single_datapoint, show=False, save_name=DIR + "Ground-Truth")

# Transform to coefficients
single_datapoint_coeffs = flatten_coeffs(sh_transform(single_datapoint[None,...]))

# Plot training model input (coefficients zeroed after L)
single_datapoint_coeffs[:,L+1:] = 0
input_datapoint = inv_sh_transform(unflatten_coeffs(single_datapoint_coeffs))
plot_tensor_as_map(input_datapoint.squeeze(), show=False, save_name=DIR + "Training-model-input")

# Plot inference model input (only coefficients up to L)
single_datapoint_coeffs = single_datapoint_coeffs[:,:L]
input_datapoint = inv_sh_transform(unflatten_coeffs(single_datapoint_coeffs))
plot_tensor_as_map(input_datapoint.squeeze(), show=False, save_name=DIR + "Inference-model-input")


# Load model checkpoint
from Transformer import LightningModel

path = f"autoregressive-downcasting/{model_str}/checkpoints/best-model.ckpt"

model = LightningModel.load_from_checkpoint(path)
model.eval()
model.to(DEVICE)

means = torch.load("data/wind-speed_level-500_testset_means.pt", weights_only=True).to(DEVICE)
stds = torch.load("data/wind-speed_level-500_testset_stds.pt", weights_only=True).to(DEVICE)

# Infer and plot prediction
model_input  = (single_datapoint_coeffs - means[:L]) / stds[:L]
raw_pred = model.infer(torch.view_as_real(model_input))
print(raw_pred.shape)
T = 60*121
rescaled_pred = torch.view_as_complex(raw_pred.squeeze()) * stds[:T] + means[:T]
pred = inv_sh_transform(unflatten_coeffs(rescaled_pred[None,:]))
plot_tensor_as_map(pred.squeeze(), show=False, save_name=DIR + "Inference-model-output")
