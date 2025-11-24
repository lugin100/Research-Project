from Plotting import *
from Dataset import WeatherDataset
from Functions import *

natural_ds = WeatherDataset("wind_speed", time_slice=slice("1970-01-01", "1970-01-02", None), level=500)


# Plot single datapoint as example
single_datapoint = natural_ds.__getitem__(4)
plot_tensor_as_map(single_datapoint, show=False, save_name="Ground-Truth")

# Transform to coefficients
single_datapoint_coeffs = flatten_coeffs(sh_transform(single_datapoint[None,...])).squeeze()

# Mask coeffs to be predicted
L = int(60*61/2)
single_datapoint_coeffs[L+1:] = 0

# Plot training model input (coefficients zeroed after L)
input_datapoint = inv_sh_transfrom(unflatten_coeffs(single_datapoint_coeffs[None,...]))
plot_tensor_as_map(input_datapoint.squeeze(), show=False, save_name="Training-model-input")

# Plot inference model input (only coefficients up to L)
single_datapoint_coeffs = single_datapoint_coeffs[:L]
input_datapoint = inv_sh_transfrom(unflatten_coeffs(single_datapoint_coeffs[None,...]))
plot_tensor_as_map(input_datapoint.squeeze(), show=False, save_name="Inference-model-input")