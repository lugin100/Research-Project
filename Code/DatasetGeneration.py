from Dataset import WeatherDataset, CoeffDataset

trainset_path = "data/wind-speed_level-500_trainset"
validationset_path = "data/wind-speed_level-500_validationset"
testset_path = "data/wind-speed_level-500_testset"

wds = WeatherDataset("wind_speed", slice("1959", "2005", None), 500)
print("Train samples: ", len(wds)) # 68664
CoeffDataset(path=trainset_path, weatherDataset=wds)

wds = WeatherDataset("wind_speed", slice("2006", "2012", None), 500)
print("Validation samples: ", len(wds)) # 10228
CoeffDataset(path=validationset_path, weatherDataset=wds)

wds = WeatherDataset("wind_speed", slice("2013", "2022", None), 500)
print("Test samples: ", len(wds)) # 13148
CoeffDataset(path=testset_path, weatherDataset=wds)

