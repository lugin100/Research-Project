## Transformer Model Training Configuation ##

RUN_NAME = "Test"

############ Setup ###############
import torch
torch.set_float32_matmul_precision("medium") # Faster on tensor cores

############ Model ###############

from Transformer import *

triangular_number = lambda N: int(N*(N+1)/2)

params = {
	"N": 60 ,			# Maximal coefficient degree in training
	"PI": 8,  			# Number of predicted mixture components
	"EPS": 1e-5, 		# Clamping constant for variance of predicted distriutions
	"D": 512,  			# Embedding dimension
	"H": 4,  			# Number of heads in multi-head attention
	"NORM_FIRST": True, # Whether to apply layer norm first or after attention and feedforward
	"LR": 5e-3, 		# Learning Rate
    "MAX_EPOCHS:" 50   # Maximal number of training epochs
}

params["T"] = triangular_number(params["N"]) # Number of coefficients given in training
params["L"] = triangular_number(params["N"]/2), # Number of coefficients given in inference

transformer = TransformerModel(**params)
model = LightningModel(transformer)


############# Datasets ##############
from Dataset import CoeffDataset
from torch.utils.data import DataLoader

params["B"] = 100  # Training batch size

ds = CoeffDataset("data/wind-speed_level-500_trainset", index_limit=params["T"])
trainloader = DataLoader(ds, batch_size=params["B"], num_workers=7, shuffle=True)

ds = CoeffDataset("data/wind-speed_level-500_testset", index_limit=params["T"])
validloader = DataLoader(ds, batch_size=params["B"], num_workers=7, shuffle=False) # No need to shuffle testset


############# Optimizer #############
from types import MethodType

params["LR"] = 5e-3 # Learning Rate
params["MAX_EPOCHS"] = 50

optimizer = torch.optim.AdamW(model.parameters(), lr=parms["LR"])
configure_optimizers = lambda self: optimizer
model.configure_optimizers = MethodType(configure_optimizers, model)


############# Logging ###############
from lightning.pytorch.loggers import WandbLogger

wandb_logger = WandbLogger(project="autoregressive-downcasting", name=RUN_NAME, log_model="all")
wandb_logger.experiments.config.update(params)
############# Execution #############
from lightning import Trainer


trainer = Trainer(max_epochs=params["MAX_EPOCHS"], logger=wandb_logger, log_every_n_steps=1)
trainer.fit(model, trainloader, validloader)
