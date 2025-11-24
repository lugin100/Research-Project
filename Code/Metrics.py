import torch

from Functions import *


def mean_squared_error(a, b):
	# Computes MSE between (optionally batched) 2d real inputs a and b
	a = torch.atleast_3d(a)
	b = torch.atleast_3d(b)
	return torch.nn.MSELoss(reduction="mean")(a, b)


def beta_nll(pis, means, variances, targets, beta=0):
	"""
    Compute beta-adjusted negative log likelihood of gaussian mixture models at given values
    Args:
        pis: Mixture components of shape (..., PI)
        means: Means of mixture components of shape (..., PI, 2)
        variances: Variances of mixture components of shape (..., PI, 2)
        targets: Where to evaluate the mixture components, shape (..., 2)
      	beta: Beta hyperparameter. 0 (default) equals no beta-correction, i.e. normal nll loss
    Returns:
        loss: Loss of targets given the mixture model, summed over leading axes
    """
    targets = targets.unsqueeze(-2) # (..., 1, 2) to broadcast over components
    # Compute nll for each component and dimension
    gaussian_nlls = 0.5 * (((targets - means)**2 / variances) + variances.log())     # (..., PI, 2)

    # Apply beta correction if needed
    if beta>0:
        # Detach variances to avoid backprop
        gaussian_nlls = gaussian_nlls * variances.detach() ** self.BETA

    # Independent dimensions -> Sum nlls over dimension
    gaussian_nlls = gaussian_nlls.sum(axis=-1) # (..., PI)

    components = pis.log() - gaussian_nlls

    # Use logsumexp for numerical stability
    mixture_nll = -torch.logsumexp(components, dim=-1) # (...)

    # Average over all remaining axes
    return mixture_nll.mean()


def nll(pis, means, variances, targets):
	return beta_nll(pis, means, variances, targets)


def pi_median(pis):
	return pis.median(-1)[0].mean()


def variance_median(variances):
	return variances.median(-1)[0].median(-1)[0].mean()



def generate_metrics():
	T = int(60*61/2)
	L = int(30*31/2)
	B = 10000
	ds = CoeffDataset("data/wind-speed_level-500_testset", index_limit=T)
	loader = DataLoader(ds, batch_size=B, num_workers=7, shuffle=False)

	path = "TODO"
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

		mse_zeros.append(mean_squared_error(weather, torch.zeros_like(weather)))
		mse_noise.append(mean_squared_error(weather, torch.randn_like(weather)))
		raw_pred = model.infer(model_input)
		pred = inv_sh_transform(unflatten_coeffs(torch.view_as_complex(raw_pred)))
		mse_preds.append(mean_squared_error(weather, pred))

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
		pi_medians.append(pi_median(pis))
		mean_mse.append(mean_squared_error(means, torch.zeros_like(means)))
		variance_medians.append(variance_median(variances))

	print("Median of mixture components avaraged over all predictions: ", sum(pi_medians)/len(pi_medians))
	print("Squared deviation of predicted means from 0, averaged over all predictions: ", sum(mean_mse)/len(mean_mse))
	print("Median of variances averaged over all predictions: ", sum(variance_medians)/len(variance_medians))

	nll_true = []
	nll_zeros = []
	nll_means = []
	nll_noise = []

	for batch in loader:
		model_input = torch.view_as_real(batch)
		params = model.coerce_params(model.forward(model_input))
		nll_true.append(nll(*params, model_input))
		nll_zeros.append(nll(*params, torch.zeros_like(model_input)))
		nll_means.append(nll(*params, params[1]))
		nll_noise.append(nll(*params, torch.randn_like(model_input)))


	print("NLL of true data with predicted parameters: ", sum(nll_true)/len(nll_true))
	print("NLL of zeros with predicted parameters: ", sum(nll_zeros)/len(nll_zeros))
	print("NLL of means with predicted parameters: ", sum(nll_means)/len(nll_means))
	print("NLL of N(0,1) with predicted parameters: ", sum(nll_noise)/len(nll_noise))