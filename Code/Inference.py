import os
import torch
from types import MethodType
from Transformer import LightningModel
from lightning import Trainer
from Dataset import CoeffDataset
from torch.utils.data import DataLoader
from Functions import (
	DEVICE,
	triangular_number,
	unflatten_coeffs,
	inv_sh_transform)
from types import MethodType


torch.set_float32_matmul_precision("medium") # Faster on tensor cores

model_str = "mitogrw5"
DIR = f"Results/{model_str}/Predictions"

# Load dataset
data_path = "data/wind-speed_level-500_testset"
testset = CoeffDataset(data_path)

# Modify dataset such that it returns index too
def __getitem__(self, idx):
	if self.index_limit is None:
		return idx, self.data[idx]
	else:
		return idx, self.data[idx, :self.index_limit]
testset.__getitem__ = MethodType(__getitem__, testset)

B = 5
dl = DataLoader(testset, batch_size=B, num_workers=7, shuffle=False)


# Load model checkpoint
model_path = f"autoregressive-downcasting/{model_str}/checkpoints/best-model.ckpt"

model = LightningModel.load_from_checkpoint(model_path)
trainer = Trainer(
    accelerator="gpu",
    devices="auto",  # Uses all available GPUs
    strategy="ddp",  # Distributed Data Parallel
)
L = triangular_number(60)
T = triangular_number(120)
os.makedirs(os.path.dirname(DIR), exist_ok=True)
os.makedirs(os.path.dirname(DIR + "/coeffs"), exist_ok=True)
os.makedirs(os.path.dirname(DIR + "/reals"), exist_ok=True)

def predict_step(self, batch, batch_idx):
	del batch_idx # This index is just a lie when using DDP
	global_indices, batch = batch
	batch = torch.view_as_real(batch[:,:L])
	pred_batch = self.infer(batch)
	rescaled_preds = torch.view_as_complex(pred_batch) * stds + means
	reals = inv_sh_transform(unflatten_coeffs(rescaled_preds))

	for batch_idx, global_idx in enumerate(global_indices):
		torch.save(pred_batch[batch_idx],
			f"{DIR}/coeffs/sample_{global_idx}.pt")
	torch.save(reals[batch_idx],
		f"{DIR}/reals/sample_{global_idx}.pt")
	

model.predict_step = MethodType(predict_step, model)

means = torch.load("data/wind-speed_level-500_testset_means.pt", weights_only=True)[None,:T].to(DEVICE)
stds = torch.load("data/wind-speed_level-500_testset_stds.pt", weights_only=True)[None,:T].to(DEVICE)
print(means.shape)

trainer.predict(model, dataloaders=dl)
