"""This module contains code for the spectral embedding based on the adjacency matrix"""

import rsvd
from scipy import sparse
from lime.Base import NodeEmbedding
from lime import utils


class AdjacencySpectralEmbedding(NodeEmbedding):
    def __init__(
        self, verbose=False,
    ):
        self.in_vec = None  # In-vector
        self.out_vec = None  # Out-vector

    def fit(self, net):
        A = utils.to_adjacency_matrix(net)
        self.A = A
        return self

    def update_embedding(self, dim):
        u, s, v = rsvd.rSVD(self.A, dim=dim)
        self.in_vec = u @ sparse.diags(s)
