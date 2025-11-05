import torch
from torch.utils.data import Dataset
from Code.Dataset import WeatherDataset
from Code.Functions import sh_transform, flatten_coeffs

class CoeffDataset(Dataset):

    def __init__(self, path):
        try:
            self.data = torch.load(path + ".pt")
        except:
            # load weather dataset and calculate coeffs
            print("Dataset not found, calculating from WeatherDataset")
            weatherData = WeatherDataset("wind_speed", slice("1970", "1971"))
            data = sh_transform(weatherData.data.mT)
            data = flatten_coeffs(data)

            # Standardization
            coeff_means = torch.mean(data, axis=-1)
            data -= coeff_means[None,:]
            torch.save(coeff_means, path + "_means.pt")
            coeff_stds = torch.std(data, axis=-1)
            data /= coeff_stds[None,:]
            torch.save(coeff_stds, path + "_stds.pt")

            torch.save(data, path + ".pt")
            self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


#coeffs = CoeffDataset("data/coeffs_wind_speed_level=500_time=slice('1970', '1972').pt")

