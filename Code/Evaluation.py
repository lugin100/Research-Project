
import torch

from Metrics import RMSE
from glob import glob
from Functions import triangular_number, inv_sh_transform, unflatten_coeffs
from Dataset import WeatherDataset, CoeffDataset
from torch.utils.data import DataLoader


model_str = "mitogrw5"
data_path = "data/wind-speed_level-500_testset"
log_path = f"Results/{model_str}/Evaluation.txt"

pred_path = f"Results/{model_str}/Predictions/reals/"

pred_files = sorted(glob(pred_path + "batch_*.pt"))

B = 5
#ground_truth_ds = WeatherDataset("wind_speed", time_slice=slice("2011", "2022", None), level=500)
ground_truth_ds = CoeffDataset("data/wind-speed_level-500_testset")
ground_truth_dl = DataLoader(ground_truth_ds, batch_size=B, num_workers=7, shuffle=False)

print(len(ground_truth_dl))
print(len(pred_files))
#assert len(ground_truth_dl) == len(pred_files)

T = triangular_number(120)
with open(log_path, "w") as log_file:
	with torch.no_grad():

		rmses = []

		for pred_path, gt_batch in zip(pred_files, ground_truth_dl):
			pred_batch = torch.load(pred_path)
			if pred_batch.shape[0] != B:
				break

			gt_batch = gt_batch[:,:T]
			gt_batch = unflatten_coeffs(gt_batch)
			gt_reals_batch = inv_sh_transform(gt_batch)
			assert pred_batch.shape == gt_reals_batch.shape

			rmse = RMSE(pred_batch, gt_reals_batch, feature_dims=(-1,-2), reduce=True)
			rmse = rmse.cpu().item()
			rmses.append(rmse)

		print("RMSE between test weather dataset and predictions: ", sum(rmses)/len(rmses), file=log_file)
