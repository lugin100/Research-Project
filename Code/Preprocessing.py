import xarray as xr
import torch_harmonics as th
import torch

default_path = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"


def download_data(level, timepoint, variable, path=None):
    """
    Download ERA5 data at select level, timepoint and variable
    and save it locally
    """
    if path is None:
        path = default_path
    data_view = xr.open_zarr(path)
    selection = data_view.sel(level=level, time=timepoint)[variable]
    
    save_file = f"data/{variable}_level={level}_time={timepoint}.zarr"
    selection.to_zarr(save_file, mode="w-", zarr_format=2, consolidated=False)

def load_data(level, timepoint, variable):
    """
    Load ERA5 data at select level, timepoint and variable
    Look up local storage; if not available, download first
    """
    save_file = f"data/{variable}_level={level}_time={timepoint}.zarr"
    try:
        data = xr.open_zarr(save_file, zarr_format=2, consolidated=False)
    except:
        print("load_data: Falling back to download, might take a while")
        download_data(level, timepoint, variable)
        data = xr.open_zarr(save_file, zarr_format=2, consolidated=False)
    finally:
        return data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def sh_transform(data):
    data = torch.as_tensor(data.to_dataarray().values).squeeze()
    (nlat, nlon) = data.shape
    sht = th.RealSHT(nlat, nlon, grid="equiangular").to(device)
    return sht(data.to(device))

def inv_sh_transfrom(coeffs):
    (lmax, mmax) = coeffs.shape
    inv_sht = th.InverseRealSHT(lmax, 2*mmax-1, grid="equiangular").to(device)
    return inv_sht(coeffs.to(device))