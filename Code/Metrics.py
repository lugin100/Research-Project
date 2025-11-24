import torch



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

