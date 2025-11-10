import xarray as xr
import torch
from Code.Functions import DEVICE

DOWNLOAD_PATH = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"



def download_data(level, time, variable, path=None):
    """
    Download ERA5 data at select level, time and variable
    and save it locally
    """
    return DeprecationWarning("Only for local use")
    path = path if path is not None else DOWNLOAD_PATH
    
    data_view = xr.open_zarr(path)
    selection = data_view.sel(level=level, time=time)[variable]
    
    save_file = f"data/{variable}_level={level}_time={time}.nc"
    selection.to_netcdf(save_file, mode="w")

def load_data(level, time, variable):
    """
    Load ERA5 data at select level, time and variable
    Look up local storage; if not available, download first
    """
    return DeprecationWarning("Only for local use")
    save_file = f"data/{variable}_level={level}_time={time}.nc"
    try:
        data = xr.open_dataarray(save_file)
    except:
        print("load_data: Falling back to download, might take a while")
        download_data(level, time, variable)
        data = xr.open_dataarray(save_file)
    finally:
        return data

class WeatherDataset(torch.utils.data.Dataset):

    DATA_PATH = "/mnt/lustre/work/ludwig/shared_datasets/weatherbench2/global1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr"

    def __init__(self, variable, time_slice, level, path = None, standardize=False):
        path = path if path is not None else self.DATA_PATH
        try:
            data = xr.open_dataset(path, engine="zarr")
            print(f"Loading dataset from {path}")
        except:
            print("Could not find dataset!")
            return
        data = data[variable]
        data = data.sel(time = time_slice, level=level)
        self.data = data.values
        self.materialize()
	if standardize:
            self.standardize()


    def materialize(self):
        '''
        Materialize lazy loaded xarray into pytorch tensor on DEVICE
        '''
        self.data = torch.as_tensor(self.data).to(DEVICE)

    def standardize(self):
        self.means = self.data.mean(0, keepdim=True)
        self.stds = self.data.std(0, keepdim=True)
        self.data = (self.data - self.means) / self.stds

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx].mT

