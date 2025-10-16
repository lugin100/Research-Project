import xarray as xr

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

