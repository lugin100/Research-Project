import torch
from torch.utils.data import Dataset
from Code.Dataset import WeatherDataset
from Code.Functions import sh_transform

class CoeffDataset(Dataset):

    def __init__(self, path):
        try:
            self.data = torch.load(path)
        except:
            # load weather dataset and calculate coeffs
            print("Dataset not found, calculating from WeatherDataset")
            weatherData = WeatherDataset("wind_speed", slice("1970", "1971"))
            self.data = sh_transform(weatherData.data.mT)
            torch.save(self.data, path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


#coeffs = CoeffDataset("data/coeffs_wind_speed_level=500_time=slice('1970', '1972').pt")

