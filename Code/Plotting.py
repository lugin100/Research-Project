from matplotlib import pyplot as plt
import xarray as xr
import torch
import numpy as np
from torch_harmonics.plotting import plot_sphere

plt.rcParams.update({'font.size': 16})

def plot_io(show=False, save_name=None):
    plt.tight_layout()
    if save_name is not None:
        path = save_name + ".pdf"
        print(path)
        plt.savefig(path, bbox_inches = "tight")
    if show:
        plt.show()
    plt.close('all') # Clear existing state in plt

def plot_xarray_as_map(data, show=True, save_name=None):
    assert isinstance(data, xr.DataArray)
    data.transpose().plot()
    plot_io(show, save_name)

def plot_tensor_as_map(data, globe=False, show=True, save_name=None, vmin=None, vmax=None):
    assert isinstance(data, torch.Tensor)
    data = data.cpu().numpy()
    if globe is False:
        plt.imshow(data, cmap = "turbo", vmin=vmin, vmax=vmax)
        plt.colorbar()
    else:
        plot_sphere(data, cmap = "turbo", colorbar=True)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plot_io(show, save_name)

def plot_coeffs_as_img(data, show=True, save_name=None):
    assert isinstance(data, torch.Tensor)
    data = data.cpu().numpy()
    plt.imshow(data, vmin=0, vmax=1.2)
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
    plt.stairs(means, label="Mean", linewidth=0.5)
    plt.fill_between(i, means - stds, means + stds, alpha=0.5, linewidth=0, label="Standard Deviation")
    plt.yscale("log")
    plt.xlabel("Flat coefficient index")
    plt.ylabel("Absolute value of coefficient")
    plt.legend()
    plot_io(show, save_name)

