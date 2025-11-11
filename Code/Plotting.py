from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import xarray as xr
import torch
import numpy as np

plt.rcParams.update({'font.size': 16})

def plot_io(show=True, save_name=None):
    plt.tight_layout()
    if save_name is not None:
        path = "Figures/" + save_name + ".pdf"
        print(path)
        plt.savefig(path, bbox_inches = "tight")
    if show:
        plt.show()

def plot_xarray_as_map(data, show=True, save_name=None):
    assert isinstance(data, xr.DataArray)
    data.transpose().plot()
    plot_io(show, save_name)

def plot_tensor_as_map(data, show=True, save_name=None):
    assert isinstance(data, torch.Tensor)
    data = data.cpu().numpy()
    plt.imshow(data)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")    
    plt.colorbar()
    plot_io(show, save_name)

def plot_coeffs_as_img(data, show=True, save_name=None):
    assert isinstance(data, torch.Tensor)
    data = data.cpu().numpy()
    plt.imshow(data, norm=LogNorm())
    plt.colorbar()
    plot_io(show, save_name)

def plot_coeffs_as_hist(data, show=True, save_name=None):
    assert isinstance(data, torch.Tensor)
    data = data.cpu().numpy()
    plt.hist(data.flatten(), log=True)
    plot_io(show, save_name)

def plot_coeffs_over_m(data, show=True, save_name=None):
    assert isinstance(data, torch.Tensor)
    data = data.cpu().numpy()
    plt.plot(data.mean(axis=0))
    plot_io(show, save_name)

def plot_coeff_stats(path, show=True, save_name=None):
    means = torch.load(path + "_means.pt", weights_only=True).abs()
    stds = torch.load(path + "_stds.pt", weights_only=True).abs()
    i = np.arange(len(means))
    plt.plot(i, means)
    plt.fill_between(i, means - stds, means + stds, alph=0.5, linewidth=0)
    plt.bar()
    plt.yscale("log")
    plt.xlabel("Flat coefficient index")
    plt.ylabel("Absolute mean and standard deviation")
    plot_io(show, save_name)

