
"""This module contains code for the spectral embedding based on the modularity matrix"""

import rsvd
from scipy import sparse 
from lime.Base import NodeEmbedding
from lime import utils

class ModularitySpectralEmbedding(NodeEmbedding):
    def __init__(
        self, verbose=False,
    ):
        self.in_vec = None  # In-vector
        self.out_vec = None  # Out-vector

    def fit(self, net):
        A = utils.to_adjacency_matrix(net)
        self.A = A
        self.deg = np.array(A.sum(axis=1)).reshape(-1)
        return self

    def update_embedding(self, dim):
        Q = [
            [self.A],
            [-self.deg.reshape((-1, 1)) / np.sum(self.deg), self.deg.reshape((1, -1))],
        ]
        u, s, v = rsvd.rSVD(Q, dim=dim)
        self.in_vec = u @ sparse.diags(s)
        self.out_vec = None

