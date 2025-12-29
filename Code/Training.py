############ Setup ###############
RUN_NAME = "V1"

import torch
from Functions import triangular_number
from Transformer import TransformerModel
torch.set_float32_matmul_precision("medium") # Faster on tensor cores

from Dataset import CoeffDataset
from torch.utils.data import DataLoader

from types import MethodType
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

############ Model ###############
L = 61 # Maximal input coefficient degree
N = 121 # Maximal output coefficient degree
params = {
	"L": triangular_number(L),  # Number of input coefficients
	"T": triangular_number(N),  # Number of output coefficients
	"D": 512,  					# Embedding dimension
	"H": 8,  					# Number of heads in multi-head attention
	"R": 4, 					# Number of sequential transformer blocks
	"PI": 8,  					# Number of predicted mixture components
	"EPS": 1e-5, 				# Clamping constant for variance of predicted distribution
	"BETA": 0.5, 				# Parameter for beta-corrected NLL loss
	"DROPOUT_PROB": 0.1			# Dropout probability for training
}

model = TransformerModel(**params)


############# Datasets ##############

params["B"] = 32  # Training and eval batch size

ds = CoeffDataset("data/wind-speed_level-500_trainset")
trainloader = DataLoader(ds, batch_size=params["B"], num_workers=8, shuffle=True)

ds = CoeffDataset("data/wind-speed_level-500_validationset")
validationloader = DataLoader(ds, batch_size=params["B"], num_workers=8, shuffle=False)


############# Optimizer #############

params.update({
	"BASE_LR": 5e-5,	 		# Base Learning Rate
	"LR_START_FACTOR": 1e-7,	# LR factor for first warmup step
    "MAX_EPOCHS": 20,    		# Maximal number of training epochs
    "WARMUP_STEPS": 200, 	    # Number of steps until warmup completion
	"MIN_LR": 1e-7				# Learning rate at last cosine step
	})

optimizer = torch.optim.AdamW(model.parameters(), lr=params["BASE_LR"])

total_steps =  params["MAX_EPOCHS"] * len(trainloader)
cosine_steps = total_steps - params["WARMUP_STEPS"]

warmup_schedule = LinearLR(
	optimizer,
	start_factor=params["LR_START_FACTOR"],
	end_factor=1,
	total_iters=params["WARMUP_STEPS"])
cosine_schedule = CosineAnnealingLR(
	optimizer,
	T_max=cosine_steps,
	eta_min=params["MIN_LR"]
	)
scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_schedule, cosine_schedule],
            milestones=[params["WARMUP_STEPS"]]
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


params["EARLY_STOPPING_PATIENCE"] = 15
early_stop = EarlyStopping(
	monitor="val/NLL Loss",
	mode="min",
	patience=params["EARLY_STOPPING_PATIENCE"],
	min_delta=0.0,
	verbose=True
	)

############# Logging ###############

wandb_logger = WandbLogger(project="autoregressive-downcasting", name=RUN_NAME, log_model="all")
wandb_logger.experiment.config.update(params)

checkpointing = ModelCheckpoint(
    monitor="val/NLL Loss",
    mode="min",
    save_top_k=1,
    filename="best-model",
	)

############# Execution #############

trainer = Trainer(
	fast_dev_run=False,
	max_epochs=params["MAX_EPOCHS"],
	logger=wandb_logger,
	log_every_n_steps=5,
	callbacks=[early_stop, checkpointing],
	val_check_interval=0.25, # Run validation more often
	)

trainer.fit(model, trainloader, validationloader)
