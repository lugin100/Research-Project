import torch

from Functions import *


def mean_squared_error(a, b):
	# Computes MSE between (optionally batched) 2d real inputs a and b
	a = torch.atleast_3d(a)
	b = torch.atleast_3d(b)
	return torch.nn.MSELoss(reduction="mean")(a, b)


def beta_nll(pis, means, variances, targets, beta=0, average=True):
	"""
	Compute beta-adjusted negative log likelihood of gaussian mixture models at given values
	Args:
	pis: Mixture components of shape (..., PI)
	means: Means of mixture components of shape (..., PI, 2)
	variances: Variances of mixture components of shape (..., PI, 2)
	targets: Where to evaluate the mixture components, shape (..., 2)
	  	beta: Beta hyperparameter. 0 (default) equals no beta-correction, i.e. normal nll loss
	  	average: Whether to average over leading axes
	Returns:
	loss: Loss of targets given the mixture model, averaged over leading axes if average=True
	"""
	targets = targets.unsqueeze(-2) # (..., 1, 2) to broadcast over components
	# Compute nll for each component and dimension
	gaussian_nlls = 0.5 * (((targets - means)**2 / variances) + variances.log())	 # (..., PI, 2)

	# Apply beta correction if needed
	if beta>0:
		# Detach variances to avoid backprop
		gaussian_nlls = gaussian_nlls * variances.detach() ** beta

	# Independent dimensions -> Sum nlls over dimension
	gaussian_nlls = gaussian_nlls.sum(axis=-1) # (..., PI)

	components = pis.log() - gaussian_nlls

	# Use logsumexp for numerical stability
	mixture_nll = -torch.logsumexp(components, dim=-1) # (...)

	# Average over all remaining axes
	return mixture_nll.mean()


def nll(pis, means, variances, targets, average=True):
	return beta_nll(pis, means, variances, targets, beta=0, average=average)


def pi_median(pis):
	return pis.median(-1)[0].mean()


def variance_median(variances):
	return variances.median(-1)[0].median(-1)[0].mean()



def generate_metrics():
	with torch.no_grad():
		T = int(60*61/2)
		L = int(30*31/2)
		B = 10000
		ds = CoeffDataset("data/wind-speed_level-500_testset", index_limit=T)
		loader = DataLoader(ds, batch_size=B, num_workers=7, shuffle=False)

		path = "Research-Project/autoregressive-downcasting/vsn5fk32/checkpoints/best-model.ckpt"
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
