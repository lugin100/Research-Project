import torch.nn.functional
from torch import nn
from torch.distributions import MixtureSameFamily

from Code.CoeffDataset import CoeffDataset
from Code.Functions import *
from torch.utils.data import DataLoader

import lightning as L

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



class TransformerModel(nn.Module):

    def __init__(self, T: int, L: int, D: int, H: int, PI: int, eps: float, **kwargs):
        super().__init__()
        self.embedding = nn.Linear(2, D)
        self.transformer = torch.nn.TransformerEncoderLayer(d_model=D, nhead=H, batch_first=True, **kwargs)
        self.mask = create_mask(T, L).to(DEVICE)
        self.unembedding = torch.nn.Linear(D, 5*PI)
        self.L = L
        self.T = T
        self.PI = PI
        self.eps = eps


    def forward(self, input):
        #input.shape # (B, T, 2)
        input = self.embedding(input) # (B, T, D)
        output = self.transformer(input, src_mask=self.mask) # (B, T, D)
        gmm_params = self.unembedding(output) # (B, T, 5*PI)
        return gmm_params


    def create_gmm(self, params):
        """
        Creates a Gaussian Mixture Model from given parameters
        :param params: of shape (B, 5*PI) stacked parameters extracted as:
        pi = params[:, 0:PI]
        mu_0 = params[:, PI:2*PI]
        mu_1 = params[:, 2*PI:3*PI]
        sigma_0 = params[:, 3*PI:4*PI]
        sigma_1 = params[:, 4*PI:5*PI]
        :return: Pytorch Gaussian Mixture Model from given parameters
        """
        PI = self.PI
        pi = params[:, 0:PI] # (B, PI)
        pi = nn.Softmax(dim=-1)(pi) # Normalize, ensure positive
        mu_0 = params[:, PI:2*PI]
        mu_1 = params[:, 2*PI:3*PI]
        mu = torch.stack([mu_0, mu_1], dim=2) # (B, PI, 2)
        sigma_0 = params[:, 3*PI:4*PI]
        sigma_1 = params[:, 4*PI:5*PI]
        sigma = torch.stack([sigma_0, sigma_1], dim=2) # (B, PI, 2)
        sigma = nn.Softplus()(sigma)
        sigma = nn.Threshold(self.eps, self.eps)(sigma)
        mix = torch.distributions.Categorical(pi)
        gaussians = torch.distributions.LowRankMultivariateNormal(mu,torch.tensor(0), sigma) # (B,PI,2)
        gaussians = torch.distributions.Independent(gaussians, 1)
        gmm = MixtureSameFamily(mix, gaussians)
        return gmm


    def infer(self, input):
        # input.shape # (B, L, 2)
        output = torch.zeros((self.B, self.T, 2), device=DEVICE)
        output[:,:self.L,:] = input
        for index in range(self.L, self.T):
            gmm_params = self.forward(output) # (B, T, 5*PI)
            gmm = self.create_gmm(gmm_params[:,index, :]) # B 2-dimensional GMMs
            sample = gmm.sample([1]) # (B, 2)
            output[:,index,:] = sample
        return output


    def loss_fn(self, gmms, values):
        return - gmms.log_prob(values)


class LightningModel(L.LightningModule):

    def __init__(self, transformer):
        super().__init__()
        self.model = transformer

    def training_step(self, batch, batch_idx):
        coeffs = torch.view_as_real(batch)
        output = model.forward(coeffs).flatten(0,1) # (B*T,5*PI)
        gmms = model.create_gmm(output)
        loss = model.loss_fn(gmms, coeffs)
        return loss

    def validation_step(self, batch, batch_idx):
        coeffs = torch.view_as_real(batch)
        output = model.forward(coeffs).flatten(0,1) # (B*T,5*PI)
        gmms = model.create_gmm(output)
        loss = model.loss_fn(gmms, coeffs)
        self.log("test_loss", loss)






#test_loss_fn = torch.nn.MSELoss(reduction="mean")



"""
def evaluate():
    model.eval()
    with torch.no_grad():
        test_loss = 0
        for batch.to(DEVICE) in test_loader:
            coeffs = torch.view_as_real(flatten_coeffs(batch))
            input = coeffs[:,:L,2]
            output = model.infer(coeffs)
            test_loss += test_loss_fn(coeffs, output).item()
        test_loss /= len(test_loader)
        print(test_loss)


def train():
    for epoch in range(EPOCHS):
        model.train()
        print(f"Epoch {epoch+1}\n")
        for batch.to(DEVICE) in dataloader:
            coeffs = torch.view_as_real(flatten_coeffs(batch))
            output = model.forward(coeffs).flatten(0,1) # (B*T,5*PI)
            gmms = model.create_gmm(output)
            loss = train_loss_fn(gmms, coeffs)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        evaluate()
"""