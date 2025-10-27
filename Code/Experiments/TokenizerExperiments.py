from typing import Callable

from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from Code.CoeffDataset import CoeffDataset
from Code.Functions import *
from Code.Tokenizer import AutoencoderTokenizer, ClusteringTokenizer

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


train_data = torch.view_as_real(next(iter(dataloader)))
train_data = flatten_coeffs(train_data).reshape((-1,2))
tokenizer = ClusteringTokenizer(train_data, V=500)

reals = train_data[...,0].cpu()
imags = train_data[...,1].cpu()
ys = tokenizer.tokenize(train_data)
#ys = model.encode_hard(coeff_batch).argmax(dim=-1).cpu()
plt.scatter(reals, imags, marker="x", alpha=0.5, c=ys, cmap="tab20")
plt.xlabel("Re")
plt.ylabel("Im")
plt.colorbar()
plt.show()

def cluster_fn(input):
    indices = tokenizer.tokenize(input.cpu())
    return torch.tensor(tokenizer.detokenize(indices)).to(DEVICE)

evaluate(cluster_fn)