import torch
from Plotting import *
from Dataset import WeatherDataset
from Functions import *
from Transformer import LightningModel

natural_ds = WeatherDataset("wind_speed", time_slice=slice("1970-01-01", "1970-01-02", None), level=500)


# Plot single datapoint as example
single_datapoint = natural_ds.__getitem__(4)
plot_tensor_as_map(single_datapoint, show=False, save_name="Ground-Truth")

# Transform to coefficients
single_datapoint_coeffs = flatten_coeffs(sh_transform(single_datapoint[None,...]))

# Mask coeffs to be predicted
L = int(60*61/2)
single_datapoint_coeffs[:,L+1:] = 0

# Plot training model input (coefficients zeroed after L)
input_datapoint = inv_sh_transform(unflatten_coeffs(single_datapoint_coeffs))
plot_tensor_as_map(input_datapoint.squeeze(), show=False, save_name="Training-model-input")

# Plot inference model input (only coefficients up to L)
single_datapoint_coeffs = single_datapoint_coeffs[:L]
input_datapoint = inv_sh_transform(unflatten_coeffs(single_datapoint_coeffs))
plot_tensor_as_map(input_datapoint.squeeze(), show=False, save_name="Inference-model-input")


# Load model checkpoint
#path = "autoregressive-downcasting/vsn5fk32/checkpoints/best-model.ckpt"
path = "autoregressive-downcasting/6ybji6ws/checkpoints/best-model.ckpt"
from Transformer import *

triangular_number = lambda N: int(N*(N+1)/2)

params = {
	"N": 60,			# Maximal coefficient degree in training
	"PI": 8,  			# Number of predicted mixture components
	"EPS": 1e-5, 		# Clamping constant for variance of predicted distriutions
	"D": 512,  			# Embedding dimension
	"H": 4,  			# Number of heads in multi-head attention
	"R": 2, 			# Number of sequential transformer blocks
	"NORM_FIRST": True, # Whether to apply layer norm first or after attention and feedforward
	"BETA": 0.5 		# Parameter for beta-corrected NLL loss
}

params["T"] = triangular_number(params["N"]) # Number of coefficients given in training
params["L"] = triangular_number(params["N"]/2) # Number of coefficients given in inference

#checkpoint = torch.load(path)
transformer = TransformerModel(**params)
model = LightningModel.load_from_checkpoint(path, transformer=transformer, strict=False)
print(model)
model.eval()
model.freeze()
model.to(DEVICE)

# Infere and plot prediction
raw_pred = model.infer(torch.view_as_real(single_datapoint_coeffs))
pred = inv_sh_transform(unflatten_coeffs(torch.view_as_complex(raw_pred)))
plot_tensor_as_map(pred.squeeze(), show=False, save_name="Inference-model-output")
