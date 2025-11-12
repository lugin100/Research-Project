## Transformer Model Training Configuation ##

############ Setup ###############
import torch
torch.set_float32_matmul_precision("medium") # Faster on tensor cores

############ Model ###############

from Transformer import *

N = 60 # Maximal coefficient degree in training
triangular_number = lambda N: int(N*(N+1)/2)
T = triangular_number(N)   # Number of coefficients given in training
L = triangular_number(N/2) # Nmber of coefficients given in inference
PI = 8  # Number of predicted mixture components
EPS = 1e-5 # Clamping constant for variance of predicted distriutions
D = 512  # Embedding dimension
H = 4  # Number of heads in multi-head attention
NORM_FIRST = True # Whether to apply layer normfirst or after attention and feedforward

transformer = TransformerModel(T=T, L=L, D=D, H=H, PI=PI, eps=EPS, norm_first=NORM_FIRST)
model = LightningModel(transformer)


############# Datasets ##############
from Dataset import CoeffDataset
from torch.utils.data import DataLoader

B = 100  # Training batch size
ds = CoeffDataset("data/wind-speed_level-500_trainset")
trainloader = DataLoader(ds, batch_size=B, shuffle=True, num_workers=7)

ds = CoeffDataset("data/wind-speed_level-500_testset")
validloader = DataLoader(ds, batch_size=B, shuffle=False, num_workers=7) # No need to shuffle testset

############# Optimizer #############
from types import MethodType
LR = 5e-3 # Learning Rate

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
configure_optimizers = lambda self: optimizer
model.configure_optimizers = MethodType(configure_optimizers, model)

EPOCHS = 10




############# Execution #############
from lightning import Trainer

trainer = Trainer()
trainer.fit(model, trainloader, validloader)
