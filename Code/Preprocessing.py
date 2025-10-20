import xarray as xr
import torch_harmonics as th
import torch

DEFAULT_PATH = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def download_data(level, timepoint, variable, path=None):
    """
    Download ERA5 data at select level, timepoint and variable
    and save it locally
    """
    if path is None:
        path = DEFAULT_PATH
    
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

class WeatherDataset(torch.utils.data.Dataset):

    def __init__(self, variable, time_slice, level=500, path = None):
        path = path if path is not None else DEFAULT_PATH
        path += variable + ".nc"
        try:
            print("Loading dataset from {path}")
            data = xr.open_dataarray(path)
        except:
            print("Could not find dataset!")
            return
        self.data = data.sel(time = time_slice, level=level)
        print(data)
        materialize()
        print(data.shape)
        normalize()

    def materialize(self):
        '''
        Materialize lazy loaded xarray into pytorch tensor
        '''
        self.data = torch.as_tensor(self,data.values)

    def standardize(self):
        self.means = self.data.mean(0, keepdim=True)
        self.stds = self.data.std(0, keepdim=True)
        self.data = (self.data - self.means) / self.stds

    def __len__(self):
        return len(self)

    def __getitem__(self, idx):
        return data[idx]


def sh_transform(data):
    (nlat, nlon) = data.shape
    sht = th.RealSHT(nlat, nlon, grid="equiangular").to(DEVICE)
    return sht(data.to(DEVICE))

def inv_sh_transfrom(coeffs):
    (lmax, mmax) = coeffs.shape
    inv_sht = th.InverseRealSHT(lmax, 2*mmax-1, grid="equiangular").to(DEVICE)
    return inv_sht(coeffs.to(DEVICE))