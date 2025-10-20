from matplotlib import pyplot as plt
import xarray as xr
import torch
from matplotlib.colors import LogNorm

plt.rcParams.update({'font.size': 16})

def plot_io(show, save_name):
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
