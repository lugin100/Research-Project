from Dataset import WeatherDataset
from Plotting import *
from Functions import *
import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

def run():
    LEVEL = 500
    TIMEPOINT = slice("1970", "1972")
    VARIABLE = "wind_speed"

    dataset = WeatherDataset("wind_speed", TIMEPOINT)


    # Minibatch of 100 timepoint examples
    originals = next(iter(DataLoader(dataset, batch_size=100, shuffle=True)))
    _coeffs = sh_transform(originals)

    loss_fn = torch.nn.MSELoss(reduction="mean")
    losses = []
    for i in range(122):
        coeffs = _coeffs.clone()
        coeffs[...,i:,:] = 0
        predictions = inv_sh_transfrom(coeffs)
        loss = loss_fn(originals, predictions).cpu().numpy()
        losses.append(loss)

    plt.plot(losses)
    plt.title("MSE Loss of reconstruction of wind speed maps\n from SH coefficients up to degree l")
    plt.xlabel("Maximal available coefficient degree l")
    plt.ylabel("MSE Loss")
    plot_io(show=True, save_name="ReconstructionImportancesOfL")
