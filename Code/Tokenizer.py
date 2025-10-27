import numpy as np
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch import nn

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


def parse_array(array):
    if isinstance(array, torch.Tensor):
        return array.cpu().numpy()
    if isinstance(array, np.ndarray):
        return array
    else:
        raise TypeError("Must pass np array or torch tensor")


class ClusteringTokenizer:

    def __init__(self, train_data, V: int):
        self.clustering = KMeans(n_clusters=V).fit(train_data.cpu())

    def tokenize(self, coeffs):
        coeffs = parse_array(coeffs).astype(np.float16)
        return self.clustering.predict(coeffs)


    def detokenize(self, tokens):
        tokens = parse_array(tokens)
        return self.clustering.cluster_centers_[tokens]


