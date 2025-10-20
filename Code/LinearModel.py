from torch import nn
from Preprocessing import *
from torch.utils.data import DataLoader

dim_in = 60
dim_out = 121

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LinearModel(nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.linear = nn.Linear(dim_in*dim_in, dim_out*dim_out, dtype=torch.complex64)

    def forward(self, coeffs_in):
        coeffs_in = nn.Flatten()(coeffs_in)
        coeffs_out = self.linear(coeffs_in)
        coeffs_out = coeffs_out.reshape((-1, self.dim_out, self.dim_out))
        pred = inv_sh_transfrom(coeffs_out)
        return pred

model = LinearModel(dim_in, dim_out).to(DEVICE)
model.train()

LEARN_RATE = 5e-3
BATCH_SIZE = 1000
EPOCHS = 40

dataset = WeatherDataset("wind_speed", slice("1970", "1971"))
print(dataset.__len__())
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

loss_fn = torch.nn.MSELoss(reduction="mean")

optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)


testset = WeatherDataset("wind_speed", slice("1971", "1972"))
test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)


def evaluate():
    model.eval()
    with torch.no_grad():
        test_loss = 0
        for batch in test_loader:
            originals = batch
            coeffs = sh_transform(batch)
            _input = coeffs[:,:dim_in, :dim_in]
            pred = model(_input)
            test_loss += loss_fn(pred, originals).item()
        test_loss /= len(test_loader)
        print(test_loss)


# Train Loop
for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}\n")
    for i, batch in enumerate(dataloader):
        originals = batch
        coeffs = sh_transform(batch)
        _input = coeffs[:,:dim_in, :dim_in]
        pred = model(_input)
        loss = loss_fn(pred, originals)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    evaluate()



