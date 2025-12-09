import torch
from types import MethodType
from Transformer import LightningModel
from lightning import Trainer
from Dataset import CoeffDataset
from torch.utils.data import DataLoader
from Functions import (
	DEVICE,
	unflatten_coeffs,
	inv_sh_transform)

torch.set_float32_matmul_precision("medium") # Faster on tensor cores

model_str = "mitogrw5"
DIR = f"Results/{model_str}/Predictions"

# Load dataset
data_path = "data/wind-speed_level-500_testset"
testset = CoeffDataset(data_path)
B = 4
dl = DataLoader(testset, batch_size=B, num_workers=7, shuffle=False)

# Load model checkpoint
model_path = f"autoregressive-downcasting/{model_str}/checkpoints/best-model.ckpt"

model = LightningModel.load_from_checkpoint(model_path)
trainer = Trainer(
    accelerator="gpu",
    devices="auto",  # Uses all available GPUs
    strategy="ddp",  # Distributed Data Parallel
)
L = 30*61

def predict_step(self, batch, batch_idx):
	batch = torch.view_as_real(batch[:,:L])
	pred_batch = self.infer(batch)
	torch.save(pred_batch, f"{DIR}/coeffs/batch_{batch_idx}.pt")
	rescaled_preds = torch.view_as_complex(pred_batch) * stds + means
	reals = inv_sh_transform(unflatten_coeffs(rescaled_preds))
	torch.save(reals, f"{DIR}/reals/batch_{batch_idx}.pt")

model.predict_step = MethodType(predict_step, model)

means = torch.load("data/wind-speed_level-500_testset_means.pt", weights_only=True).to(DEVICE)
stds = torch.load("data/wind-speed_level-500_testset_stds.pt", weights_only=True).to(DEVICE)


for i, pred_batch in enumerate(trainer.predict(model, dataloaders=dl)):
	
