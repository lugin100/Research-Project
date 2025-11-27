import torch
from torch.utils.data import DataLoader

from Transformer import LightningModel
from matplotlib import pyplot as plt
from Functions import *
from Dataset import CoeffDataset
from Plotting import plot_io
from Metrics import *
with torch.no_grad():
		T = int(60*61/2)
		L = int(30*31/2)
		B = 100
		ds = CoeffDataset("data/wind-speed_level-500_testset", index_limit=T)
		loader = DataLoader(ds, batch_size=B, num_workers=7, shuffle=False)

		#path = "autoregressive-downcasting/vsn5fk32/checkpoints/best-model.ckpt"
		path = "autoregressive-downcasting/6ybji6ws/checkpoints/best-model.ckpt"
		model = LightningModel.load_from_checkpoint(path)
		model.eval()
		model.freeze()
		model.to(DEVICE)

		mse_zeros = []
		mse_noise = []
		mse_preds = []

		for batch in loader:
			weather = inv_sh_transform(unflatten_coeffs(batch))
			model_input = torch.view_as_real(batch[:,:L])

			mse_zeros.append(mean_squared_error(weather, torch.zeros_like(weather)).item())
			mse_noise.append(mean_squared_error(weather, torch.randn_like(weather)).item())
			raw_pred = model.infer(model_input)
			pred = inv_sh_transform(unflatten_coeffs(torch.view_as_complex(raw_pred)))
			mse_preds.append(mean_squared_error(weather, pred).item())

		print("MSE between test weather dataset and zeros: ", sum(mse_zeros)/len(mse_zeros))
		print("MSE between test weather dataset and N(0,1) noise: ", sum(mse_noise)/len(mse_noise))
		print("MSE between test weather dataset and predictions: ", sum(mse_preds)/len(mse_preds))

		pi_medians = []
		mean_mse = []
		variance_medians = []

		for batch in loader:
			model_input = torch.view_as_real(batch)
			pred_params = model.coerce_params(model.forward(model_input))
			pis, means, variances = params
			pi_medians.append(pi_median(pis).item())
			mean_mse.append(mean_squared_error(means, torch.zeros_like(means)).item())
			variance_medians.append(variance_median(variances).item())

		print("Median of mixture components avaraged over all predictions: ", sum(pi_medians)/len(pi_medians))
		print("Squared deviation of predicted means from 0, averaged over all predictions: ", sum(mean_mse)/len(mean_mse))
		print("Median of variances averaged over all predictions: ", sum(variance_medians)/len(variance_medians))


		# NLL with predicted parameters
		nll_true = []
		nll_zeros = []
		nll_means = []
		nll_noise = []

		for batch in loader:
			model_input = torch.view_as_real(batch)
			params = model.coerce_params(model.forward(model_input))
			nll_true.append(nll(*params, model_input).item())
			nll_zeros.append(nll(*params, torch.zeros_like(model_input)).item())
			nll_means.append(nll(*params, params[1]).item())
			nll_noise.append(nll(*params, torch.randn_like(model_input)).item())


		print("NLL of true data with predicted parameters: ", sum(nll_true)/len(nll_true))
		print("NLL of zeros with predicted parameters: ", sum(nll_zeros)/len(nll_zeros))
		print("NLL of means with predicted parameters: ", sum(nll_means)/len(nll_means))
		print("NLL of N(0,1) with predicted parameters: ", sum(nll_noise)/len(nll_noise))

		# NLL with inferred parameters
		nll_true = torch.zeros(T)
		n = 0

		for batch in loader:
			B = batch.shape[0]
			n += B
			model_input = torch.view_as_real(batch)
			output = torch.zeros_like(model_input)
			output[:,:L,:] = model_input[:,L:]
			for index in range(self.L, self.T):
				preds = self.forward(output) # (B, T, 5*PI)
				params = self.coerce_parameters(preds)
				gmm = self.create_gmms(*params[:,index,:]) # B 2-dimensional GMMs
				sample = gmm.sample([1]) # (B, 2)
				output[:,index,:] = sample


			losses = nll(*params, model_input, average=False) # (B,T)
			nll_true += losses.sum(axis=0).cpu() # (T)

		plt.figure()

		plt.plot(nll_true)
		plt.ylabel("NLL of true data")
		plt.xlabel("Inferred parameter index")
		plot_io(show=False, save_name="NLL-for-inferred-params")
