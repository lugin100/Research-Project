import torch.nn.functional
from torch import nn
from torch.distributions import MixtureSameFamily, Categorical, Normal, Independent

from Functions import *
import Metrics

from lightning import LightningModule

'''
class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int, n_heads: int, dropout_prob: float = 0.0):
        super().__init__()
        self.q_proj = nn.Linear(input_dim, embedding_dim, bias=True)
        self.k_proj = nn.Linear(input_dim, embedding_dim, bias=True)
        self.v_proj = nn.Linear(input_dim, embedding_dim, bias=True)

        self.out_proj = nn.Linear(embedding_dim, input_dim, bias=True)
        assert embedding_dim % n_heads == 0, "Embedding dim is not divisible by nheads"
        self.head_dim = embedding_dim // n_heads

    def forward(self, query, key, value):
        query = self.q_proj(query)
        query = query.unflatten(-1, [self.n_heads, self.head_dim]).transpose(1, 2)
        key = self.k_proj(key)
        key = key.unflatten(-1, [self.n_heads, self.head_dim]).transpose(1, 2)
        value = self.v_proj(value)
        value = value.unflatten(-1, [self.n_heads, self.head_dim]).transpose(1, 2)
        attention = torch.nn.functional.scaled_dot_product_attention(query, key, value, dropout_p=self.dropout_prob)
        attention = attention.transpose(1, 2).flatten(-2)
        return self.out_proj(attention)
'''

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
        nn.init.kaiming_normal_(layer.weight, mode="fan_out")
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
        self.transformer = torch.nn.TransformerEncoder(transformer_layer, num_layers=R) # initializes all identically -> reinitialize afterwards
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
        output = torch.zeros((self.B, self.T, 2), device=DEVICE)
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

    def __init__(self, transformer):
        super().__init__()
        self.model = transformer
        self.logs = []


    def infer(self, input):
        return self.transformer.infer(input)


    def training_step(self, batch, batch_idx):
        coeffs = torch.view_as_real(batch)
        output = self.model.forward(coeffs) # (B,T,5*PI)
        params = self.model.coerce_parameters(output)
        loss = self.model.beta_nll_loss(*params, coeffs)
        self.log("Beta NLL Loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return loss


    def validation_step(self, batch, batch_idx):
        coeffs = torch.view_as_real(batch)
        output = self.model.forward(coeffs) # (B,T,5*PI)
        params = self.model.coerce_parameters(output)
        pis, means, variances = params
        loss = self.model.beta_nll_loss(*params, coeffs)
        pi_log = Metrics.pi_median(pis)
        mse_log = Metrics.mean_squared_error(means, torch.tensor(0.))
        variance_log = Metrics.variance_median(variances)
        self.logs.append({"pi_log": pi_log, "mse_log": mse_log, "variance_log": variance_log, "loss_log": loss})


    def on_validation_epoch_end(self):
        pi_log = torch.stack([x["pi_log"] for x in self.logs]).mean()
        mse_log = torch.stack([x["mse_log"] for x in self.logs]).mean()
        variance_log = torch.stack([x["variance_log"] for x in self.logs]).mean()
        loss_log = torch.stack([x["loss_log"] for x in self.logs]).mean()
        self.log("val/PI Median", pi_log, prog_bar=True, logger=True, on_epoch=True, add_dataloader_idx=False)
        self.log("val/MSE", mse_log, prog_bar=True, logger=True, on_epoch=True, add_dataloader_idx=False)
        self.log("val/Variance Median", variance_log, prog_bar=True, logger=True, on_epoch=True, add_dataloader_idx=False)
        self.log("val/NLL Loss", loss_log, prog_bar=True, logger=True, on_epoch=True, add_dataloader_idx=False)
        self.logs.clear()
