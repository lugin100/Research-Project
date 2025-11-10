from Dataset import *

testset_path = "./data/wind-speed_level-500_testset"
trainset_path = ".data/wind-speed_level-500_trainset"

wds = WeatherDataset("wind_speed", slice("1959", "2010", None), 500)
CoeffDataset(path=trainset_path, weatherDataset=wds)

wds = WeatherDataset("wind_speed", slice("2011", "2022", None), 500)
CoeffDataset(path=testset_path, weatherDataset=wds)
