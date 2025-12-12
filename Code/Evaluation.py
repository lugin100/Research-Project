from pathlib import Path
import re
import torch

from Metrics import RMSE
from glob import glob
from Functions import DEVICE, triangular_number, inv_sh_transform, unflatten_coeffs
from Dataset import CoeffDataset
from torch.utils.data import DataLoader
from Plotting import plot_coeffs_as_img

model_str = "interpolation"
data_path = "data/wind-speed_level-500_testset"
log_path = f"Results/{model_str}/"

pred_path = f"Results/{model_str}/Predictions/"

def sort(file):
	return int(re.search(r"batch_(\d+)\.pt", file.name).group(1))

real_pred_files = Path(pred_path).glob("reals/batch_*.pt")
real_pred_files = sorted(real_pred_files, key=sort)

coeff_pred_files = Path(pred_path).glob("coeffs/batch_*.pt")
coeff_pred_files = sorted(coeff_pred_files, key=sort)

B = 256
#ground_truth_ds = WeatherDataset("wind_speed", time_slice=slice("2011", "2022", None), level=500)
ground_truth_ds = CoeffDataset("data/wind-speed_level-500_testset")
ground_truth_dl = DataLoader(ground_truth_ds, batch_size=B, num_workers=7, shuffle=False)

#print(len(ground_truth_dl))
#print(len(pred_files))
#assert len(ground_truth_dl) == len(pred_files)

T = triangular_number(121)

means = torch.load("data/wind-speed_level-500_testset_means.pt", weights_only=True)[None,:T].to(DEVICE)
stds = torch.load("data/wind-speed_level-500_testset_stds.pt", weights_only=True)[None,:T].to(DEVICE)

with open(log_path + "Evaluation.txt", "w") as log_file:
	with torch.no_grad():

		# RMSE of real data prediction
		rmses = []
		for pred_path, gt_batch in zip(real_pred_files, ground_truth_dl):
			pred_batch = torch.load(pred_path, weights_only=True).to(DEVICE)
			if pred_batch.shape[0] != B:
				break
			gt_batch = gt_batch.to(DEVICE)
			gt_batch = gt_batch[:,:T] * stds + means
			gt_batch = unflatten_coeffs(gt_batch)
			gt_reals_batch = inv_sh_transform(gt_batch)
			assert pred_batch.shape == gt_reals_batch.shape

			rmse = RMSE(pred_batch, gt_reals_batch, feature_dims=(-1,-2), reduce=True)
			rmse = rmse.cpu().item()
			rmses.append(rmse)

		print("RMSE between test weather dataset and predictions: ", sum(rmses)/len(rmses), file=log_file)

		# RMSE of predicted coefficients
		i = 0
		rmses = torch.zeros(T)
		for pred_path, gt_batch in zip(coeff_pred_files, ground_truth_dl):
			pred_batch = torch.load(pred_path, weights_only=True).cpu()
			if pred_batch.shape[0] != B:
				break
			if i>1:
				break
			gt_batch = torch.view_as_real(gt_batch[:,:T]).cpu()
			print((pred_batch - gt_batch)[:2,1810:1850,:])
			assert pred_batch.shape == gt_batch.shape
			rmses_batch = RMSE(pred_batch, gt_batch, feature_dims=-1, reduce=False)
			rmses += rmses_batch.mean(axis=0)
			i += 1
		rmses = rmses / i

		rmses = unflatten_coeffs(rmses.unsqueeze(0)).squeeze()
		plot_coeffs_as_img(rmses, show=False, save_name=log_path + "Coefficient_RMSE")

		rmse = rmses.mean()
		print("RMSE between predicted coefficients and ground truth: ", rmse.item(), file=log_file)
