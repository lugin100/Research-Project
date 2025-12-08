import torch
from torch.utils.data import DataLoader

from Transformer import LightningModel
from matplotlib import pyplot as plt
from Functions import (
	DEVICE,
	inv_sh_transform,
	unflatten_coeffs)
from Dataset import CoeffDataset
from Plotting import plot_io
from Metrics import (
	mean_squared_error,
	pi_median,
	variance_median,
	nll)


with torch.no_grad():
		T = int(60*61/2)
		L = int(30*31/2)
		B = 100
		ds = CoeffDataset("data/wind-speed_level-500_testset", index_limit=T)
		loader = DataLoader(ds, batch_size=B, num_workers=7, shuffle=False)

#		path = "autoregressive-downcasting/6ybji6ws/checkpoints/best-model.ckpt"
		path = "autoregressive-downcasting/mu4ikkcc/checkpoints/best-model.ckpt"
		model = LightningModel.load_from_checkpoint(path).model
		model.eval()
		model.to(DEVICE)

		mse_zeros = []
		mse_noise = []
		mse_preds = []
		mse_coeff_noise = []

		means = torch.load("data/wind-speed_level-500_testset_means.pt", weights_only=True)[:T].to(DEVICE)
		stds = torch.load("data/wind-speed_level-500_testset_stds.pt", weights_only=True)[:T].to(DEVICE)

		for i, batch in enumerate(loader):
			print(i)
			if i > 1:
				break
			rescaled_batch = batch.to(DEVICE) * stds[None,:] + means[None,:]
			weather = inv_sh_transform(unflatten_coeffs(rescaled_batch))
			
			
#			model_input = torch.view_as_real(batch[:,:L])
#			raw_pred = torch.view_as_complex(model.infer(model_input))
#			rescaled_pred = raw_pred * stds[None,:] + means[None,:]
#			pred = inv_sh_transform(unflatten_coeffs(rescaled_pred))

			noise = torch.randn_like(batch).to(DEVICE)
			scaled_noise = noise * stds[None,:] + means[None,:]
			weather_noise = inv_sh_transform(unflatten_coeffs(scaled_noise))
			mse_zeros.append(mean_squared_error(weather, torch.zeros_like(weather)).item())
			mse_noise.append(mean_squared_error(weather, torch.randn_like(weather)).item())
			mse_coeff_noise.append(mean_squared_error(weather, weather_noise).item())
#			mse_preds.append(mean_squared_error(weather, pred).item())

		print("MSE between test weather dataset and zeros: ", sum(mse_zeros)/len(mse_zeros))
		print("MSE between test weather dataset and N(0,1) noise: ", sum(mse_noise)/len(mse_noise))
		print("MSE between test weather dataset and transformed N(0,1) coefficient noise: ", sum(mse_coeff_noise)/len(mse_coeff_noise))
#		print("MSE between test weather dataset and predictions: ", sum(mse_preds)/len(mse_preds))

		pi_medians = []
		mean_mse = []
		variance_medians = []

		for i, batch in enumerate(loader):
			print(i)
			if i > 10:
				break
			model_input = torch.view_as_real(batch).to(DEVICE)
			pred_params = model.coerce_parameters(model.forward(model_input))
			pis, means, variances = pred_params
			pi_medians.append(pi_median(pis).item())
			mean_mse.append(mean_squared_error(means, torch.zeros_like(means)).item())
			variance_medians.append(variance_median(variances).item())

		print("Median of mixture components avaraged over all predictions: ", sum(pi_medians)/len(pi_medians))
		print("Squared deviation of predicted means from 0, averaged over all predictions: ", sum(mean_mse)/len(mean_mse))
		print("Median of variances averaged over all predictions: ", sum(variance_medians)/len(variance_medians))


		# NLL with predicted parameters
		nll_zeros = []
		nll_means = []
		nll_noise = []
		nll_true = []

		for i, batch in enumerate(loader):
			print(i)
			if i > 10:
				break
			model_input = torch.view_as_real(batch).to(DEVICE)
			params = model.coerce_parameters(model.forward(model_input))
			means = torch.einsum("btp,btpd->btd", params[0], params[1])
			nll_zeros.append(nll(*params, torch.zeros_like(model_input)).item())
			nll_means.append(nll(*params, means).item())
			nll_noise.append(nll(*params, torch.randn_like(model_input)).item())
			nll_true.append(nll(*params, model_input).item())


		print("NLL of zeros with predicted parameters: ", sum(nll_zeros)/len(nll_zeros))
		print("NLL of means with predicted parameters: ", sum(nll_means)/len(nll_means))
		print("NLL of N(0,1) with predicted parameters: ", sum(nll_noise)/len(nll_noise))
		print("NLL of true data with predicted parameters: ", sum(nll_true)/len(nll_true))

		# NLL with inferred parameters
		nll_true = torch.zeros(T)
		n = 0

		for i, batch in enumerate(loader):
			print(i)
			if i > 1:
				break
			n += B
			model_input = torch.view_as_real(batch[:,:L]).to(DEVICE)
			output = model.infer(model_input)
			preds = model.forward(output) # (B, T, 5*PI)
			params = model.coerce_parameters(preds)
			losses = nll(*params, torch.view_as_real(batch).to(DEVICE), average=False) # (B,T)
			nll_true += losses.sum(axis=0).cpu() # (T)

		plt.figure()

		plt.plot(nll_true)
		plt.ylabel("NLL of true data")
		plt.xlabel("Inferred parameter index")
		plot_io(show=False, save_name="NLL-for-inferred-params")
