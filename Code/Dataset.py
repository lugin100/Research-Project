import xarray as xr
import torch
from torch.utils.data import Dataset, DataLoader
from Functions import *
from tqdm import tqdm


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


class WeatherDataset(Dataset):

    DATA_PATH = "/mnt/lustre/work/ludwig/shared_datasets/weatherbench2/global/1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr"

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

    def get(self):
        return self.data.mT


    def materialize(self):
        '''
        Materialize lazy loaded xarray into pytorch tensor on DEVICE
        '''
        self.data = torch.as_tensor(self.data)#.to(DEVICE)

    def standardize(self):
        self.means = self.data.mean(0, keepdim=True)
        self.stds = self.data.std(0, keepdim=True)
        self.data = (self.data - self.means) / self.stds

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx].mT


class CoeffDataset(Dataset):

    def __init__(self, path, weatherDataset=None):

        try:
            self.data = torch.load(path + ".pt", weights_only=True)

        except:
            # create coefficients from weather dataset
            assert weatherDataset is not None, "Could not find dataset at given path and no weatherDataset passed"
            batch_size = 10000
            data = []
            for batch in tqdm(DataLoader(weatherDataset, batch_size=batch_size, shuffle=False, pin_memory=True)):
                with torch.no_grad():
                    batch = sh_transform(batch.to(DEVICE, non_blocking=True))
                    batch = flatten_coeffs(batch)
                data.append(batch.detach().cpu())

            data = torch.cat(data, dim=0)
            self.data = data

            # Standardization
            coeff_means = torch.mean(data, axis=0)
            coeff_stds = torch.std(data, axis=0)
            data /= coeff_stds[None,:]
            data -= coeff_means[None,:]

            print(f"Saving dataset to {path}")
            torch.save(data, path + ".pt")
            torch.save(coeff_means, path + "_means.pt")
            torch.save(coeff_stds, path + "_stds.pt")


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        return self.data[idx]

