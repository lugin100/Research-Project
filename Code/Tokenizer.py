import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from Code.Dataset import WeatherDataset
from Code.Functions import DEVICE, sh_transform, inv_sh_transfrom, flatten_coeffs, unflatten_coeffs


class Tokenizer(nn.Module):
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



model = Tokenizer(dim_in = 2, V=500).to(DEVICE)
model.load_state_dict(torch.load('tokenizer-gumbel-weights.pth', weights_only=True))
model.train()

LEARN_RATE = 5e-3
B = 100
EPOCHS = 20
dataset = WeatherDataset("wind_speed", slice("1970", "1971"))
dataloader = DataLoader(dataset, batch_size=B, shuffle=True)

loss_fn = torch.nn.MSELoss(reduction="mean")

optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)

def evaluate():
    model.eval()
    with torch.no_grad():
        test_loss = 0
        for batch in dataloader:
            originals = batch
            coeffs = sh_transform(batch)
            _input = flatten_coeffs(coeffs)
            (B,T) = _input.shape
            _input = torch.view_as_real(_input).reshape(B * T, 2)
            one_hot = model.encode_hard(_input).float()
            output = model.decode(one_hot)
            output = torch.view_as_complex(output).reshape(B, T)
            output = unflatten_coeffs(output)
            pred = inv_sh_transfrom(output)
            test_loss += loss_fn(pred, originals).item()
        test_loss /= len(dataloader)
        print(test_loss)
    model.train()

def train():
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}")
        for i, batch in enumerate(dataloader):
            originals = batch
            coeffs = sh_transform(batch)
            _input = flatten_coeffs(coeffs)
            (B,T) = _input.shape
            _input = torch.view_as_real(_input).reshape(B*T, 2)
            intermediate = model.encode_soft(_input)
            output = model.decode(intermediate)
            output = torch.view_as_complex(output).reshape(B,T)
            output = unflatten_coeffs(output)
            pred = inv_sh_transfrom(output)
            loss = loss_fn(pred, originals)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        evaluate()

#train()
evaluate()
torch.save(model.state_dict(), 'tokenizer-gumbel-weights.pth')