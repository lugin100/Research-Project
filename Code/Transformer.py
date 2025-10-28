import torch.nn.functional
from torch import nn
from torch.distributions import MixtureSameFamily

from Code.CoeffDataset import CoeffDataset
from Code.Functions import *
from torch.utils.data import DataLoader

from Code.Tokenizer import ClusteringTokenizer

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

    def __init__(self, T: int, L: int, D: int, H: int, PI: int, **kwargs):
        super().__init__()
        self.embedding = nn.Linear(2, D)
        self.transformer = torch.nn.TransformerEncoderLayer(d_model=D, nhead=H, batch_first=True, **kwargs)
        self.mask = create_mask(T, L).to(DEVICE)
        self.unembedding = torch.nn.Linear(D, 5*PI)
        self.L = L
        self.T = T
        self.PI = PI


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
        :raises: ValueError if a pi is negative or a sigma is non-positive
        """
        PI = self.PI
        pi = params[:, 0:PI] # (B, PI)
        if (pi < 0).any():
            raise ValueError("pi must be non-negative")
        mu_0 = params[:, PI:2*PI]
        mu_1 = params[:, 2*PI:3*PI]
        mu = torch.stack([mu_0, mu_1], dim=2) # (B, PI, 2)
        sigma_0 = params[:, 3*PI:4*PI]
        sigma_1 = params[:, 4*PI:5*PI]
        # TODO: Clamp sigmas? softplus? softmax for pi's?
        sigma = torch.stack([sigma_0, sigma_1], dim=2) # (B, PI, 2)
        if (sigma <= 0).any():
            raise ValueError("sigma must be positive")
        mix = torch.distributions.Categorical(pi)
        gaussians = torch.distributions.LowRankMultivariateNormal(mu,torch.tensor(0), sigma) # (B,PI,2)
        gaussians = torch.distributions.Independent(gaussians, 1)
        gmm = MixtureSameFamily(mix, gaussians)
        return gmm


    def infer(self, input):
        # input.shape # (B, T, 2)
        for index in range(self.L, self.T):
            gmm_params = self.forward(input) # (B, T, 5*PI)
            gmm = self.create_gmm(gmm_params[:,index, :]) # B 2-dimensional GMMs
            sample = gmm.sample([1]) # (B, 2)
            input[:,index,:] = sample
        return input

N = 121
T = int(N*(N+1)/2)
L = int(60*61/2)
model = TransformerModel(T=T, L=L, V=500, D=512, H=4).to(DEVICE)

LEARN_RATE = 5e-3
B = 10
EPOCHS = 10
dataset = CoeffDataset("data/coeffs_wind_speed_level=500_time=slice('1970', '1972').pt")
dataloader = DataLoader(dataset, batch_size=B, shuffle=True)

loss_fn = torch.nn.MSELoss(reduction="mean")

optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)

testset = CoeffDataset("data/coeffs_wind_speed_level=500_time=slice('1970', '1972').pt")
test_loader = DataLoader(testset, batch_size=B, shuffle=True)

kmeans_train_data = torch.view_as_real(next(iter(dataloader)))
kmeans_train_data = flatten_coeffs(kmeans_train_data).reshape((-1,2))

V = 500
tokenizer = ClusteringTokenizer(train_data=kmeans_train_data, V=V)


def evaluate():
    model.eval()
    with torch.no_grad():
        test_loss = 0
        for batch in test_loader:
            coeffs = torch.view_as_real(flatten_coeffs(batch))
            tokens = tokenizer.tokenize(coeffs)
            #output = model.infer(tokens)
            output = tokens # Identity function to try out
            output = tokenizer.detokenize(output)
            test_loss += loss_fn(coeffs, torch.tensor(output).to(DEVICE)).item()
        test_loss /= len(test_loader)
        print(test_loss)


def train():
    for epoch in range(EPOCHS):
        model.train()
        print(f"Epoch {epoch+1}\n")
        for i, batch in enumerate(dataloader):
            batch = batch.to(DEVICE)
            input = flatten_coeffs(batch)
            output = model.forward(input)
            loss = loss_fn(output, input)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        evaluate()

evaluate()
