import os
import torch
from types import MethodType
from Transformer import TransformerModel
from lightning import Trainer
from lightning.pytorch.callbacks import BasePredictionWriter
from Dataset import CoeffDataset
from torch.utils.data import DataLoader
from Functions import (
	DEVICE,
	triangular_number,
	unflatten_coeffs,
	inv_sh_transform)


torch.set_float32_matmul_precision("medium") # Faster on tensor cores

model_str = "0u0y7zwn"
checkpoint = ""
DIR = f"Results/{model_str}{checkpoint}/Predictions"
L = triangular_number(61)
T = triangular_number(121)
# Load dataset
data_path = "data/wind-speed_level-500_testset"
testset = CoeffDataset(data_path, index_limit=L)

# Modify dataset such that it returns index too
def __getitem__(self, idx):
	if self.index_limit is None:
		return torch.tensor(idx), self.data[idx]
	else:
		return torch.tensor(idx), self.data[idx, :self.index_limit]
testset.__getitem__ = MethodType(__getitem__, testset)

B = 256
dl = DataLoader(testset, batch_size=B, num_workers=7, shuffle=False)


# Load model checkpoint
model_path = f"autoregressive-downcasting/{model_str}/checkpoints/best-model{checkpoint}.ckpt"

model = TransformerModel.load_from_checkpoint(model_path)

os.makedirs(os.path.dirname(DIR), exist_ok=True)
os.makedirs(os.path.dirname(DIR + "/coeffs"), exist_ok=True)
os.makedirs(os.path.dirname(DIR + "/reals"), exist_ok=True)

means = torch.load("data/wind-speed_level-500_testset_means.pt", weights_only=True)[None,:T].to(DEVICE)
stds = torch.load("data/wind-speed_level-500_testset_stds.pt", weights_only=True)[None,:T].to(DEVICE)

def predict_step(self, batch, batch_idx):
	input = torch.view_as_real(batch[:,:L])
	pred_batch = self.infer(input)
	rescaled_preds = torch.view_as_complex(pred_batch) * stds + means
	reals = inv_sh_transform(unflatten_coeffs(rescaled_preds))
	return pred_batch, reals

model.predict_step = MethodType(predict_step, model)

class PredictionWriter(BasePredictionWriter):

	def __init__(self):
		super().__init__("batch")

	def write_on_batch_end(self, trainer, model, predictions, batch_indices, batch, batch_idx, dataloader_idx):
		coeffs, reals = predictions
		torch.save(coeffs, DIR + f"/coeffs/batch_{batch_idx}.pt")
		torch.save(reals, DIR + f"/reals/batch_{batch_idx}.pt")

predWriter = PredictionWriter()
trainer = Trainer(
    devices="auto",
    strategy="ddp",
    callbacks=[predWriter]
)

trainer.predict(model, dataloaders=dl, return_predictions=False)
