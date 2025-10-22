import xarray as xr
import torch
import Functions

DOWNLOAD_PATH = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"



def download_data(level, time, variable, path=None):
    """
    Download ERA5 data at select level, time and variable
    and save it locally
    """
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

    LOCAL_PATH = "data/wind_speed_level=500_time=slice('1970', '1972', None)"
    
    def __init__(self, variable, time_slice, level=500, path = None):
        path = path if path is not None else self.LOCAL_PATH
        path += ".nc"
        try:
            print(f"Loading dataset from {path}")
            data = xr.open_dataarray(path)
        except:
            print("Could not find dataset!")
            return
        #self.data = data.sel(time = time_slice, level=level)
        self.data = data
        #print(data)
        self.materialize()
        #print(data.shape)
        self.standardize()

    def materialize(self):
        '''
        Materialize lazy loaded xarray into pytorch tensor
        '''
        self.data = torch.as_tensor(self.data.values).to(Functions.DEVICE)

    def standardize(self):
        self.means = self.data.mean(0, keepdim=True)
        self.stds = self.data.std(0, keepdim=True)
        self.data = (self.data - self.means) / self.stds

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx].mT

