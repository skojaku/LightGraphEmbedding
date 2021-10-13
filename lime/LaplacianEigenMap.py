"""This module contains code for Laplacian Eigenmaps.

Mikhail Belkin, and Partha Niyogi. Laplacian Eigenmaps and Spectral Techniques for Embedding and Clustering. NIPS 2001.

"""

import numpy as np
from scipy import sparse
from lime import rsvd
from lime import NodeEmbedding
from lime import utils

class LaplacianEigenMap(NodeEmbedding):
    def __init__(self):
        self.in_vec = None
        self.L = None
        self.deg = None

    def fit(self, G):
        A = utils.to_adjacency_matrix(G)

        # Compute the (inverse) normalized laplacian matrix
        deg = np.array(A.sum(axis=1)).reshape(-1)
        Dsqrt = sparse.diags(1 / np.maximum(np.sqrt(deg), 1e-12), format="csr")
        L = Dsqrt @ A @ Dsqrt

        self.L = L
        self.deg = deg
        return self

    def transform(self, dim, return_out_vector=False):
        if self.in_vec is None:
            self.update_embedding(dim)
        elif self.in_vec.shape[1] != dim:
            self.update_embedding(dim)
        return self.in_vec

    def update_embedding(self, dim):
        u, s, _ = rsvd.rSVD(self.L, dim + 1)  # add one for the trivial solution
        order = np.argsort(s)[::-1][1:]
        u = u[:, order]

        Dsqrt = sparse.diags(1 / np.maximum(np.sqrt(self.deg), 1e-12), format="csr")
        self.in_vec = Dsqrt @ u
        self.out_vec = u
