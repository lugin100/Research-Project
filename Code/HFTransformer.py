import torch
from torch import nn

import Metrics
from Functions import DEVICE

from lightning import LightningModule
from transformers import GPT2Config, GPT2Model

class TransformerModel(LightningModule):

	def __init__(self, 
		T: int, 
		L: int, 
		D: int, 
		H: int, 
		R: int, 
		PI: int, 
		EPS: float, 
		BETA: float, 
		NORM_FIRST: bool, 
		DROPOUT_PROB: float):

		super().__init__()
		self.save_hyperparameters()
		self.T = T
		self.L = L
		self.BETA = BETA
		self.logs = []
		config = GPT2Config(
			n_positions = T,
			n_ctx = T,
			n_embd = D,
			n_layer = R,
			n_head = H,
			activation_function = "relu",
			attn_pdrop = DROPOUT_PROB,
			layer_norm_epsilon = EPS,
			use_cache = True,
			)
		self.transformer = GPT2Model(config)
		self.head = GMMHead(D, PI, EPS, BETA)

	def forward(self, input, cache=None):
		"""
		Model forward pass. 

		If cache=None (default), does not use caching
		and returns just the foward pass result.
		If cache=-1, returns forward pass and its cache.
		Else: Passes cache argument to be used as cache and returns 
		forward pass and new cache
		"""
		use_cache = cache is not None
		cache = None if cache == -1 else cache
		output = self.transformer(
			inputs_embeds = input,
			past = cache,
			use_cache = use_cache,
			)
		hiddens = output.last_hidden_state
		output = self.head(hiddens)
		if not use_cache:
			return output
		else:
			cache = output.past
		return output, cache


	def beta_nll_loss(self, pis, means, variances, targets):
		beta = self.BETA if self.training else 0.
		return Metrics.beta_nll(pis, means, variances, targets, beta)


	def training_step(self, batch, batch_idx):
		input = torch.view_as_real(batch)
		params = self.forward(input)
		loss = self.beta_nll_loss(*params, input)
		self.log("Beta NLL Loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
		return loss


	def validation_step(self, batch, batch_idx):
		batch = torch.view_as_real(batch)
		params = self.forward(batch)
		pis, means, variances = params
		loss = self.beta_nll_loss(*params, batch)
		pi_log = Metrics.pi_median(pis)
		mse_log = Metrics.mean_squared_error(means, torch.zeros_like(means))
		variance_log = Metrics.variance_median(variances)
		self.logs.append({"pi_log": pi_log, "mse_log": mse_log, "variance_log": variance_log, "loss_log": loss})


	def on_validation_epoch_end(self):
		pi_log = torch.stack([x["pi_log"] for x in self.logs]).mean()
		mse_log = torch.stack([x["mse_log"] for x in self.logs]).mean()
		variance_log = torch.stack([x["variance_log"] for x in self.logs]).mean()
		loss_log = torch.stack([x["loss_log"] for x in self.logs]).mean()
		self.log("val/PI Median", pi_log, prog_bar=True, logger=True, on_epoch=True, add_dataloader_idx=False, sync_dist=True)
		self.log("val/MSE", mse_log, prog_bar=True, logger=True, on_epoch=True, add_dataloader_idx=False, sync_dist=True)
		self.log("val/Variance Median", variance_log, prog_bar=True, logger=True, on_epoch=True, add_dataloader_idx=False, sync_dist=True)
		self.log("val/NLL Loss", loss_log, prog_bar=True, logger=True, on_epoch=True, add_dataloader_idx=False, sync_dist=True)
		self.logs.clear()


	def infer(self, input):
		# input.shape # (B, L, 2)
		output = torch.zeros((input.shape[0], self.T, 2), device=DEVICE)
		output[:,:self.L,:] = input
		# Compute cache over input sequence
		cache = -1
		x = input
		for index in range(self.L, self.T):
			params, cache = self.forward(x, cache)
			params = [var[:,-1,...] for var in params]
			samples = self.sample_gmm(*params)
			output[:,index,:] = samples
			x = samples.unsqueeze(1)


	def sample_gmm(self, pis, means, variances):
		"""
		Sample from a GMM with the passed parameters.

		Args:
			pis: Mixture components of shape (..., PI)
			means: Means of mixture components of shape (..., PI, 2)
			variances: Variances of mixture components of shape (..., PI, 2)
		Returns:
			Samples from (batch) of GMMs of shape (..., 2)
		"""
		*batch_dims, components = pis.shape
		assert components == self.PI
		pis = pis.reshape(-1, self.PI)
		means = means.reshape(-1, self.PI, 2)
		variances = variances.reshape(-1, self.PI, 2)

		chosen_components = torch.multinomial(pis, 1)
		chosen_components = chosen_components.unsqueeze(-1).expand(-1,1,2)

		selected_means = torch.gather(means, 1, chosen_components).squeeze(1)
		selected_variances = torch.gather(variances, 1, chosen_components).squeeze(1)

		unit_samples = torch.randn_like(selected_means)
		samples = unit_samples * torch.sqrt(selected_variances) + selected_means
		return samples.reshape(*batch_dims, 2)


class GMMHead(nn.Module):

	def __init__(self, D, PI, EPS, BETA):
		self.PI = PI,
		self.EPS = EPS,
		self.unembedding = nn.Linear(D, 5*PI)

	def coerce_parameters(self, output):
		"""
		Coerce model outputs to be parameters of a GMM.
		Args:
		output: network outputs with shape (..., 5*PI)
		Returns:
		tuple of pis, means and variances:
		pis: mixture component weights of shape (...,PI), normalized with softmax
		means: means of gaussians of shape (..., PI, 2)
		variances: variances of gaussians of shape (..., PI, 2) with softplus applied and clipped to self.EPS
		"""
		PI = self.PI
		pi = output[..., 0:PI]
		# Normalize, ensure positive
		pis = nn.Softmax(dim=-1)(pi)

		mu_0 = output[..., PI:2*PI]
		mu_1 = output[..., 2*PI:3*PI]
		means = torch.stack([mu_0, mu_1], dim=-1)

		sigma_0 = output[..., 3*PI:4*PI]
		sigma_1 = output[..., 4*PI:5*PI]
		sigmas = torch.stack([sigma_0, sigma_1], dim=-1)
		sigmas = nn.Softplus()(sigmas)
		sigmas = nn.Threshold(self.EPS, self.EPS)(sigmas)

		return pis, means, sigmas

	def forward(self, input):
		output = self.unembedding(input)
		return self.coerce_parameters(output)
