import math

import torch_harmonics as th
import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sh_transform(data, requires_grad=False):
    (B, nlat, nlon) = data.shape
    context = torch.enable_grad() if requires_grad else torch.no_grad()
    with context:
        sht = th.RealSHT(nlat, nlon, grid="equiangular").to(DEVICE)
        return sht(data.to(DEVICE, non_blocking=True))


def inv_sh_transform(coeffs, requires_grad=False):
    (B, lmax, mmax) = coeffs.shape
    context = torch.enable_grad() if requires_grad else torch.no_grad()
    with context:
        inv_sht = th.InverseRealSHT(lmax, 2*(mmax-1), grid="equiangular").to(DEVICE)
        return inv_sht(coeffs.to(DEVICE, non_blocking=True))


def flatten_coeffs(coeffs):
    """
    Flatten a batch of lower triangular matrices of coefficients
        :param coeffs: (B,N,N) matrix of complex coefficients
    :return:
        (B, N*(N+1)/2) matrix of flattened coefficients
    """
    M = coeffs.shape[1]
    indices = torch.tril_indices(M, M, offset=0)
    return coeffs[:, indices[0], indices[1]]


def unflatten_coeffs(flat_coeffs):
    """
    Unflatten a batch of 1D tensors of coefficients to lower triangular matrices
        :param flat_coeffs: (B,N*(N+1)/2) matrix of complex coefficients
    :return:
        (B,N,N) matrix of unflattened coefficients
    :raises ValueError:
            if the input shape cannot be resolved to a triangular shape
    """

    B,T = flat_coeffs.shape
    N = (-1 + math.sqrt(1 + 8*T)) / 2  # inverse of Gauss formula for triangular numbers
    if N.is_integer():
        N = int(N)
    else:
        raise ValueError("The input shape cannot be resolved to a triangular shape")
    output = torch.zeros((B,N,N), device=flat_coeffs.device, dtype=flat_coeffs.dtype)
    indices = torch.tril_indices(N, N, offset=0)
    output[:, indices[0], indices[1]] = flat_coeffs
    return output