
import torch

from matplotlib import pyplot as plt
from Plotting import plot_io
from Metrics import RMSE
from glob import glob
from Dataset import WeatherDataset
from torch.utils.data import DataLoader


model_str = "interpolation"

log_path = f"Results/{model_str}/Evaluation.txt"

pred_path = f"Results/{model_str}/Predictions/reals/"
print(pred_path)
pred_files = sorted(glob(pred_path + "batch_*.pt"))

B = 256
ground_truth_ds = WeatherDataset("wind_speed", time_slice=slice("2011", "2022", None), level=500)
ground_truth_dl = DataLoader(ground_truth_ds, batch_size=B, num_workers=7, shuffle=False)

assert len(ground_truth_dl) == len(pred_files)

with open(log_path, "w") as log_file:
	with torch.no_grad():

		rmses = []

		for pred_path, ground_truth in zip(pred_files, ground_truth_dl):
			batch = torch.load(pred_path)
			if batch.shape[0] != B:
				break
			assert batch.shape == ground_truth.shape


			rmse = RMSE(batch, ground_truth, feature_dims=(-1,-2), reduce=True)
			rmse = rmse.cpu().item()
			rmses.append(rmse)

		print("RMSE between test weather dataset and predictions: ", sum(rmses)/len(rmses), file=log_file)
