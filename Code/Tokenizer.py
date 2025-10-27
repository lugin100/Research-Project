import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from torch import nn
from torch.utils.data import DataLoader
from triton.language import float16

from Code.CoeffDataset import CoeffDataset
from Code.Functions import *


class AutoencoderTokenizer(nn.Module):
    def __init__(self, dim_in, V):
        super().__init__()
        self.V = V
        self.dim_in = dim_in
        self.encoder = nn.Linear(dim_in, V)
        self.decoder = nn.Linear(V, dim_in)

    def encode_soft(self, x):
        # x.shape # (B,dim_in)
        logits = self.encoder(x) # (B,V)
        return self.gumbel_softmax(logits, hard=False)

    def encode_hard(self, x):
        # x.shape # (B,dim_in)
        logits = self.encoder(x)  # (B,V)
        indices = self.gumbel_softmax(logits, hard=True)
        #indices =  torch.argmax(logits, dim=-1)
        return F.one_hot(indices, self.V)

    def decode(self, x):
        return self.decoder(x)

    def gumbel_softmax(self, x, temperature=0.1, hard=False):
        """
        Sample from the Gumbel-Softmax distribution.
        """
        gumbel_noise = -torch.empty_like(x).exponential_().log()
        gumbel_logits = (x + gumbel_noise) / temperature
        if hard:
            indices = gumbel_logits.argmax(dim=-1)
            return indices
        else:
            return F.softmax(gumbel_logits, dim=-1)

class ClusteringTokenizer:

    def __init__(self, train_data, V: int):
        self.clustering = KMeans(n_clusters=V).fit(train_data.cpu())

    def parse_array(self, array):
        if isinstance(array, torch.Tensor):
            return array.to(torch.float16).cpu().numpy()
        if isinstance(array, np.ndarray):
            return array.astype(float)
        else:
            raise TypeError("Must pass np array or torch tensor")


    def tokenize(self, coeffs):
        coeffs = self.parse_array(coeffs)
        print(coeffs.dtype)
        return self.clustering.predict(coeffs)


    def detokenize(self, tokens):
        tokens = self.parse_array(tokens)
        return self.clustering.cluster_centers_[tokens]


