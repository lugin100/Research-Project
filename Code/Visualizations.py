from Plotting import *
from Dataset import WeatherDataset
from Functons import *

natural_ds = WeatherDataset("wind_speed", time_slice=slice("1970-01-01", "1970-01-02", None), level=500)

single_datapoint = natural_ds.__getitem__(4)
plot_tensor_as_map(single_datapoint, show=False, save_name="Ground Truth")

single_datapoint_coeffs = flatten_coeffs(sh_transform(single_datapoint))

print(single_datapoint_coeffs.shape)
# Mask coeffs to be predicted
L = 60*61/2
single_datapoint_coeffs[L:] = 0

input_datapoint = inv_sh_transform(single_datapoint_coeffs)

plot_tensor_as_map(input_datapoint, show=False, save_name="Model input")