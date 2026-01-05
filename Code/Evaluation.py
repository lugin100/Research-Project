from pathlib import Path
import re
import torch

from Metrics import RMSE
from Functions import DEVICE, triangular_number, inv_sh_transform, unflatten_coeffs, flatten_coeffs
from Dataset import CoeffDataset
from torch.utils.data import DataLoader
#from Plotting import plot_coeffs_as_img, plot_RMSE_over_time

model_str = "random"
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
ground_truth_ds = CoeffDataset(data_path)
ground_truth_dl = DataLoader(ground_truth_ds, batch_size=B, num_workers=7, shuffle=False)

assert len(ground_truth_dl) == len(coeff_pred_files), f"{len(ground_truth_dl)} =/= {len(coeff_pred_files)}"

T = triangular_number(121)

means = torch.load("data/wind-speed_level-500_testset_means.pt", weights_only=True)[None,:T].to(DEVICE)
stds = torch.load("data/wind-speed_level-500_testset_stds.pt", weights_only=True)[None,:T].to(DEVICE)

with open(log_path + "Evaluation.txt", "w") as log_file:
	with torch.no_grad():

		# RMSE of real data prediction
		rmses = []

		for pred_path, gt_batch in zip(real_pred_files, ground_truth_dl):
			pred_batch = torch.load(pred_path, weights_only=True).to(DEVICE)
			gt_batch = gt_batch.to(DEVICE)
			gt_batch = gt_batch[:,:T] * stds + means
			gt_batch = unflatten_coeffs(gt_batch)
			gt_reals_batch = inv_sh_transform(gt_batch)
			assert pred_batch.shape == gt_reals_batch.shape, f"{pred_batch.shape} =/= {gt_reals_batch.shape}"

			rmse_batch = RMSE(pred_batch, gt_reals_batch, feature_dims=(-1,-2), reduce=False)
			rmses.extend(rmse_batch.tolist())

		torch.save(torch.tensor(rmses), log_path + "RMSE_over_time.pt")
#		plot_coeffs_as_img(rmses, save_name=log_path + "RMSE_over_time")
		print("RMSE between test weather dataset and predictions: ", sum(rmses)/len(rmses), file=log_file)

		# RMSE of predicted coefficients
		rmses = torch.zeros(T)
		i = 0

		for pred_path, gt_batch in zip(coeff_pred_files, ground_truth_dl):
			pred_batch = torch.load(pred_path, weights_only=True).cpu()
			gt_batch = torch.view_as_real(gt_batch[:,:T]).cpu()
			assert pred_batch.shape == gt_batch.shape, f"{pred_batch.shape} =/= {gt_batch.shape}"
			rmses_batch = RMSE(pred_batch, gt_batch, feature_dims=-1, reduce=False)
			rmses += rmses_batch.mean(axis=0)
			i += 1

		rmses = rmses / i
		rmse = rmses.mean()
		print("RMSE between predicted coefficients and ground truth: ", rmse.item(), file=log_file)

		rmses = unflatten_coeffs(rmses.unsqueeze(0)).squeeze()
		torch.save(rmses, log_path + "Coefficient_RMSE.pt")
#		plot_coeffs_as_img(rmses, show=False, save_name=log_path + "Coefficient_RMSE")
