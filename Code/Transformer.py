import torch.nn.functional
from torch import nn
from Dataset import WeatherDataset
from Functions import *
from torch.utils.data import DataLoader
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
    mask = torch.full((T,T), 0.0)
    mask -= torch.full((T,T), float("inf")).triu(diagonal=1)
    mask[L:,L:] -= torch.full((T-L,T-L), float("inf")).triu(diagonal=0)
    return mask



class TransformerModel(nn.Module):

    def __init__(self, T: int, L: int, V: int, D: int, H, **kwargs):
        super().__init__()
        self.embedding = torch.nn.Embedding(V, D)
        self.transformer = torch.nn.TransformerEncoderLayer(d_model=D, nhead=H, batch_first=True, **kwargs)
        self.mask = create_mask(T, L)
        self.output_proj = torch.nn.Linear(D, V)
        self.L = L
        self.T = T


    def forward(self, input):
        #input.shape # (B, T)
        input = self.embedding(input) # (B, T, D)
        output = self.transformer(input, mask=self.mask) # (B, T, D)
        logits = self.output_proj(output) # (B, T, V)
        return logits


    def infer(self, input):
        for index in range(self.L, self.T):
            pred = self.forward(input)
            next_token = torch.argmax(pred[:,index], dim=-1)
            input[:,index] = next_token
        return input


model = TransformerModel().to(DEVICE)
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
