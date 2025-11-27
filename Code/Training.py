## Transformer Model Training Configuation ##

RUN_NAME = "SMall-model-training"

############ Setup ###############
import torch
torch.set_float32_matmul_precision("medium") # Faster on tensor cores

############ Model ###############

from Transformer import *

triangular_number = lambda N: int(N*(N+1)/2)

params = {
	"N": 60,			# Maximal coefficient degree in training
	"PI": 8,  			# Number of predicted mixture components
	"EPS": 1e-5, 		# Clamping constant for variance of predicted distriutions
	"D": 512,  			# Embedding dimension
	"H": 4,  			# Number of heads in multi-head attention
	"R": 2, 			# Number of sequential transformer blocks
	"NORM_FIRST": True, # Whether to apply layer norm first or after attention and feedforward
	"BETA": 0.5 		# Parameter for beta-corrected NLL loss
}

params["T"] = triangular_number(params["N"]) # Number of coefficients given in training
params["L"] = triangular_number(params["N"]/2) # Number of coefficients given in inference

model = LightningModel(**params)


############# Datasets ##############
from Dataset import CoeffDataset
from torch.utils.data import DataLoader

params["B"] = 128  # Training and eval batch size

ds = CoeffDataset("data/wind-speed_level-500_trainset", index_limit=params["T"])
trainloader = DataLoader(ds, batch_size=params["B"], num_workers=7, shuffle=True)

ds = CoeffDataset("data/wind-speed_level-500_testset", index_limit=params["T"])
validloader = DataLoader(ds, batch_size=params["B"], num_workers=7, shuffle=False) # No need to shuffle testset


############# Optimizer #############
from types import MethodType
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

params.update({
	"BASE_LR": 1e-5,	 		# Base Learning Rate
	"LR_START_FACTOR": 1e-8,	# LR factor for first warmup step
    "MAX_EPOCHS": 50,    		# Maximal number of training epochs
    "WARMUP_DURATION": 0.1, 	# Fraction of epochs until warmup completion
	"MIN_LR": 1e-7				# Learning rate at last cosine step
	})

optimizer = torch.optim.AdamW(model.parameters(), lr=params["BASE_LR"])

total_steps =  params["MAX_EPOCHS"] * len(trainloader)
warmup_steps = int(params["WARMUP_DURATION"] * total_steps)
cosine_steps = total_steps - warmup_steps

warmup_schedule = LinearLR(
	optimizer,
	start_factor=params["LR_START_FACTOR"],
	end_factor=1,
	total_iters=warmup_steps)
cosine_schedule = CosineAnnealingLR(
	optimizer,
	T_max=cosine_steps,
	eta_min=params["MIN_LR"]
	)
scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_schedule, cosine_schedule],
            milestones=[warmup_steps]
        )

def configure_optimizers(self):
	return {
		"optimizer": optimizer,
		"lr_scheduler": {
			"scheduler": scheduler,
			"interval": "step"
		}}
model.configure_optimizers = MethodType(configure_optimizers, model)

def on_before_optimizer_step(self, optimizer):
	cur_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
	cur_lr = optimizer.param_groups[0]["lr"]
	self.log("LR", cur_lr, prog_bar=True, on_step=True, on_epoch=False, logger=True, add_dataloader_idx=False)
model.on_before_optimizer_step = MethodType(on_before_optimizer_step, model)


############# Logging ###############
from lightning.pytorch.loggers import WandbLogger

wandb_logger = WandbLogger(project="autoregressive-downcasting", name=RUN_NAME, log_model="all")
wandb_logger.experiment.config.update(params)


############# Execution #############
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

params["EARLY_STOPPING_PATIENCE"] = 5
early_stop = EarlyStopping(
	monitor="val/NLL Loss",
	mode="min",
	patience=params["EARLY_STOPPING_PATIENCE"],
	min_delta=0.0,
	verbose=True
	)

checkpointing = ModelCheckpoint(
    monitor="val/NLL Loss",
    mode="min",
    save_top_k=1,
    filename="best-model",
	)

trainer = Trainer(
	fast_dev_run=False, 
	max_epochs=params["MAX_EPOCHS"], 
	logger=wandb_logger,
	log_every_n_steps=1,
	callbacks=[early_stop, checkpointing]
	)

trainer.fit(model, trainloader, validloader)

