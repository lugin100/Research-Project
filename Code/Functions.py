import torch_harmonics as th
import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def sh_transform(data):
    (N, nlat, nlon) = data.shape
    sht = th.RealSHT(nlat, nlon, grid="equiangular").to(DEVICE)
    return sht(data.to(DEVICE))

def inv_sh_transfrom(coeffs):
    (N, lmax, mmax) = coeffs.shape
    inv_sht = th.InverseRealSHT(lmax, 2*(mmax-1), grid="equiangular").to(DEVICE)
    return inv_sht(coeffs.to(DEVICE))