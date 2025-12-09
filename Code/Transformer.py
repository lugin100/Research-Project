import torch.nn.functional
from torch import nn
from torch.distributions import MixtureSameFamily, Categorical, Normal, Independent

from Functions import DEVICE
import Metrics

from lightning import LightningModule


class CachedMultiHeadAttention(nn.Module):
    def __init__(self, D: int, H: int, mask, dropout_prob: float):
        super().__init__()
        assert D % H == 0
        self.D = D
        self.d = D // H
        self.mask = mask
        self.dropout_prob = dropout_prob
        self.q_proj = nn.Linear(D, D, bias=True)
        self.k_proj = nn.Linear(D, D, bias=True)
        self.v_proj = nn.Linear(D, D, bias=True)

        self.out_proj = nn.Linear(D, D, bias=True)

    def forward(self, input, cached_k=None, cached_v=None):
        """
        Calculate multi-head self attention for input.
        Use cached keys and values for inference.

        Args:
            input: Self attention input of shape (B,T,D) during training and 
            (B,1,D) during inference.
            cached_k: Cached keys of shape (B,H,Ti,d)
            cached_v: Cached keys of shape (B,H,Ti,d)
        Returns:
            Tuple of transformer outputs of shape (B,T,D),
            cached keys and cached values both of shape (B,H,T,d) during training
            and (B,H,Ti+1,d) during inference
        """
        B,T,D = input.shape
        assert self.D == D
        
        query = self.q_proj(input)
        key = self.k_proj(input)
        value = self.v_proj(input)
        query = query.view(B, T, self.H, self.d).transpose(1, 2)
        key = key.view(B, T, self.H, self.d).transpose(1, 2)
        value = value.view(B, T, self.H, self.d).transpose(1, 2)
        # Query, key and value have shape (B,H,T,d)

        if T != 1:
            mask = self.mask
            assert mask.shape == (T, T)
        else:
            assert cached_k is not None
            assert cached_v is not None
            B,H,Ti,d = cached_k.shape
            mask = self.mask[Ti,:Ti+1][None,:]

            key = torch.cat([cached_k, key], dim=2)
            value = torch.cat([cached_v, value], dim=2)
        
        dropout_p = self.dropout_prob if self.training else 0.
        attention = torch.nn.functional.scaled_dot_product_attention(
            query, 
            key, 
            value, 
            attn_mask=mask,
            is_causal=False,
            dropout_p=dropout_p)

        attention = attention.transpose(1, 2).reshape((B,T,D))
        output = self.out_proj(attention)
        return output, key, value


def create_mask(T: int, L: int):
    '''
    Creates causal mask for attention.
     T: the total length of the sequence.
     L: the length of the input sequence.
     The first L tokens can attend to all previous tokens and themselves.
     The tokens after L can only attend to previous tokens.
    '''
    mask = torch.full((T,T), float("-inf"))
    mask[:,:L] = 0  # all coeffs can attend to all coeffs with index < L
    mask[torch.tril_indices(T,T,offset=-1)] = 0 # all coeffs can attend to all previous coeffs
    return mask


def initialize_layer(layer):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_normal_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
    else:
        # Do nothing at the moment, because only linear layers have params
        return


class TransformerModel(nn.Module):

    def __init__(self, T: int, L: int, D: int, H: int, R: int, PI: int, EPS: float, BETA: float, NORM_FIRST: bool, **kwargs):
        super().__init__()
        del kwargs
        self.embedding = nn.Linear(2, D)
        self.positional_encoding = nn.Parameter(torch.zeros(T, D))
        transformer_layer = torch.nn.TransformerEncoderLayer(d_model=D, nhead=H, batch_first=True, norm_first=NORM_FIRST)
        self.transformer = torch.nn.TransformerEncoder(transformer_layer, num_layers=R, enable_nested_tensor=False) # initializes all identically -> reinitialize afterwards
        self.mask = create_mask(T, L).to(DEVICE)
        self.unembedding = torch.nn.Linear(D, 5*PI)
        self.L = L
        self.T = T
        self.PI = PI
        self.EPS = EPS
        self.BETA = BETA
        # Initialization
        self.apply(initialize_layer)


    def forward(self, input):
        #input.shape # (B, T, 2)
        embedded = self.embedding(input) + self.positional_encoding.unsqueeze(0) # (B, T, D)
        output = self.transformer(embedded, mask=self.mask) # (B, T, D)
        output = self.unembedding(output) # (B, T, 5*PI)
        return output


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


    def create_gmms(self, pis, means, variances):
        """
        Creates a Gaussian Mixture Model from given parameters.
        Multiple leading batch dimensions are flattened to one.
        """
        pis = pis.reshape((-1,self.PI))
        means = means.reshape((-1, self.PI, 2))
        variances = variances.reshape((-1, self.PI, 2))
        mix = Categorical(pis)
        gaussians = Normal(means, variances)
        components = Independent(gaussians, 1)
        gmms = MixtureSameFamily(mix, components)
        return gmms


    def infer(self, input):
        # input.shape # (B, L, 2)
        output = torch.zeros((input.shape[0], self.T, 2), device=DEVICE)
        output[:,:self.L,:] = input
        # for all indices to be predicted
        for index in range(self.L, self.T):
            preds = self.forward(output) # (B, T, 5*PI)
            next_params = self.coerce_parameters(preds[:,index,:])
            gmm = self.create_gmms(*next_params) # B 2-dimensional GMMs
            sample = gmm.sample([1]) # (B, 2)
            output[:,index,:] = sample
        return output


    def beta_nll_loss(self, pis, means, variances, targets):
        beta = self.BETA if self.training else 0.
        return Metrics.beta_nll(pis, means, variances, targets, beta)


class LightningModel(LightningModule):

    def __init__(self, T: int, L: int, D: int, H: int, R: int, PI: int, EPS: float, BETA: float, NORM_FIRST: bool, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = TransformerModel(T, L, D, H, R, PI, EPS, BETA, NORM_FIRST, **kwargs)
        self.logs = []


    def infer(self, input):
        return self.model.infer(input)


    def training_step(self, batch, batch_idx):
        coeffs = torch.view_as_real(batch)
        output = self.model.forward(coeffs) # (B,T,5*PI)
        params = self.model.coerce_parameters(output)
        loss = self.model.beta_nll_loss(*params, coeffs)
        self.log("Beta NLL Loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        return loss


    def validation_step(self, batch, batch_idx):
        coeffs = torch.view_as_real(batch)
        output = self.model.forward(coeffs) # (B,T,5*PI)
        params = self.model.coerce_parameters(output)
        pis, means, variances = params
        loss = self.model.beta_nll_loss(*params, coeffs)
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
