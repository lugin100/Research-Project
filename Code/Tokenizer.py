from typing import Callable

import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from torch import nn
from torch.utils.data import DataLoader

from Code.CoeffDataset import CoeffDataset
from Code.Functions import *


class AutoencoderTokenizer(nn.Module):
    def __init__(self, dim_in, V):
        super().__init__()
        self.V = V
        self.dim_in = dim_in
        self.encoder = nn.Linear(dim_in, V)
        self.decoder = nn.Linear(V, dim_in)

    def encode_soft(self, x):
        # x.shape # (B,dim_in)
        logits = self.encoder(x) # (B,V)
        return self.gumbel_softmax(logits, hard=False)

    def encode_hard(self, x):
        # x.shape # (B,dim_in)
        logits = self.encoder(x)  # (B,V)
        indices = self.gumbel_softmax(logits, hard=True)
        #indices =  torch.argmax(logits, dim=-1)
        return F.one_hot(indices, self.V)

    def decode(self, x):
        return self.decoder(x)

    def gumbel_softmax(self, x, temperature=0.1, hard=False):
        """
        Sample from the Gumbel-Softmax distribution.
        """
        gumbel_noise = -torch.empty_like(x).exponential_().log()
        gumbel_logits = (x + gumbel_noise) / temperature
        if hard:
            indices = gumbel_logits.argmax(dim=-1)
            return indices
        else:
            return F.softmax(gumbel_logits, dim=-1)



model = AutoencoderTokenizer(dim_in = 2, V=500).to(DEVICE)
model.load_state_dict(torch.load('Models/tokenizer-gumbel-weights.pth', weights_only=True))
model.train()

LEARN_RATE = 5e-3
B = 50
EPOCHS = 20
dataset = CoeffDataset("data/coeffs_wind_speed_level=500_time=slice('1970', '1972').pt")
dataloader = DataLoader(dataset, batch_size=B, shuffle=True)

loss_fn = torch.nn.MSELoss(reduction="mean")

optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)

def model_fn(input):
    one_hot = model.encode_hard(input).float()
    return model.decode(one_hot)

def evaluate(f: Callable):
    model.eval()
    with torch.no_grad():
        test_loss = 0
        for batch in dataloader:
            _input = flatten_coeffs(batch)
            (B,T) = _input.shape
            _input = torch.view_as_real(_input).reshape(B * T, 2)
            originals = _input
            pred = f(_input)
            test_loss += loss_fn(pred, originals).item()
        test_loss /= len(dataloader)
        print(test_loss)
    model.train()

def train():
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}")
        for i, batch in enumerate(dataloader):
            _input = flatten_coeffs(batch)
            (B, T) = _input.shape
            _input = torch.view_as_real(_input).reshape(B * T, 2)
            originals = _input
            intermediate = model.encode_soft(_input)
            output = model.decode(intermediate)
            #pred = torch.view_as_complex(output).reshape(B, T)
            loss = loss_fn(output, originals)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        evaluate()

#train()
#evaluate()
#torch.save(model.state_dict(), 'Models/tokenizer-gumbel-weights-v2.pth')

coeff_batch = torch.view_as_real(next(iter(dataloader)))
coeff_batch = flatten_coeffs(coeff_batch).reshape((-1,2))
reals = coeff_batch[...,0].cpu()
imags = coeff_batch[...,1].cpu()

clustering = KMeans(n_clusters=500).fit(coeff_batch.cpu())
ys = clustering.labels_
#ys = model.encode_hard(coeff_batch).argmax(dim=-1).cpu()
plt.scatter(reals, imags, marker="x", alpha=0.5, c=ys, cmap="tab20")
plt.xlabel("Re")
plt.ylabel("Im")
plt.colorbar()
plt.show()

def cluster_fn(input):
    indices = clustering.predict(input.cpu())
    return torch.tensor(clustering.cluster_centers_[indices]).to(DEVICE)

evaluate(cluster_fn)