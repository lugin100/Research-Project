import torch

from matplotlib import pyplot as plt
from Plotting import plot_io
from Metrics import mean_squared_error
from glob import glob
from Dataset import WeatherDataset
from torch.utils.data import DataLoader


model_str = "mitogrw5"

log_path = f"Results/{model_str}/Evaluation.txt"

pred_path = f"Results/{model_str}/Predictions"
pred_files = sorted(glob(pred_path + "batch_*.pt"))

B = 5
ground_truth_ds = WeatherDataset("wind_speed", time_slice=slice("1970-01-01", "1970-01-02", None), level=500)
ground_truth_dl = DataLoader(ground_truth_ds, batch_size=B, num_workers=7, shuffle=False)

assert len(ground_truth_dl) == len(pred_files)
with open(log_path, "w") as log_file:
	with torch.no_grad():

		mses = []

		for pred_path, ground_truth in zip(pred_files, ground_truth_dl):
			batch = torch.load(pred_path)
			assert batch.shape == ground_truth.shape


			mses.append(mean_squared_error(batch, ground_truth))

		print("MSE between test weather dataset and predictions: ", sum(mses)/len(mses), file=log_file)