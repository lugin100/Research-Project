import math

import torch_harmonics as th
import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def sh_transform(data):
    (B, nlat, nlon) = data.shape
    sht = th.RealSHT(nlat, nlon, grid="equiangular").to(DEVICE)
    return sht(data.to(DEVICE))

def inv_sh_transfrom(coeffs):
    (B, lmax, mmax) = coeffs.shape
    inv_sht = th.InverseRealSHT(lmax, 2*(mmax-1), grid="equiangular").to(DEVICE)
    return inv_sht(coeffs.to(DEVICE))

def flatten_coeffs(coeffs):
    '''
    Flatten a batch of lower triangular matrices of coefficients
        :param coeffs: (B,M,M) matrix of complex coefficients
    :return:
        (B, M*(M+1)/2) matrix of flattened coefficients
    '''
    M = coeffs.shape[1]
    indices = torch.tril_indices(M, M, offset=0)
    return coeffs[:, indices[0], indices[1]]

def unflatten_coeffs(flat_coeffs):
    '''
    Unflatten a batch of 1D tensors of coefficients to lower triangular matrices
        :param flat_coeffs: (B,M*(M+1)/2) matrix of complex coefficients
    :return:
        (B,M,M) matrix of unflattened coefficients
    '''

    B,I = flat_coeffs.shape
    M = (-1 + math.sqrt(1 + 8*I)) / 2  # inverse of Gauss formula for triangular numbers
    if M.is_integer():
        M = int(M)
    else:
        raise ValueError("The input shape cannot be resolved to a triangular shape")
    output = torch.zeros((B,M,M), device=flat_coeffs.device, dtype=flat_coeffs.dtype)
    indices = torch.tril_indices(M, M, offset=0)
    output[:, indices[0], indices[1]] = flat_coeffs
    return output