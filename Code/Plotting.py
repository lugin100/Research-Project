from matplotlib import pyplot as plt
import xarray as xr
import torch
from matplotlib.colors import LogNorm


def plot_xarray_as_map(data, show=True, save_name=None):
    assert isinstance(data, xr.DataArray)
    data.transpose().plot()
    if save_name is not None:
        path = "Figures/" + save_name + ".pdf"
        print(path)
        plt.savefig(path)
    if show:
        plt.show()


def plot_tensor_as_map(data, show=True, save_name=None):
    assert isinstance(data, torch.Tensor)
    data = data.cpu().numpy()
    plt.imshow(data)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")    
    plt.colorbar()
    if save_name is not None:
        path = "Figures/" + save_name + ".pdf"
        print(path)
        plt.savefig(path)
    if show:
        plt.show()


def plot_coeffs_as_img(data, show=True, save_name=None):
    assert isinstance(data, torch.Tensor)
    data = data.cpu().numpy()
    plt.imshow(data, norm=LogNorm())
    plt.colorbar()
    if save_name is not None:
        path = "Figures/" + save_name + ".pdf"
        print(path)
        plt.savefig(path)
    if show:
        plt.show()

