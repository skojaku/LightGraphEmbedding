"""This module contains the code for the spectral embedding based on non-
backtracking matrix.

Florent Krzakala, Cristopher Moore, Elchanan Mossel, Joe Neeman, Allan
Sly, Lenka Zdeborová, Pan Zhang Proceedings of the National Academy of
Sciences Dec 2013, 110 (52) 20935-20940; DOI: 10.1073/pnas.1312486110.
"""

import numpy as np
from scipy import sparse

from lime import utils
from lime.Base import NodeEmbedding


class NonBackTrackingSpectralEmbedding(NodeEmbedding):
    def __init__(
        self, verbose=False,
    ):
        self.in_vec = None  # In-vector
        self.out_vec = None  # Out-vector

    def fit(self, net):
        A = utils.to_adjacency_matrix(net)
        self.deg = np.array(A.sum(axis=1)).reshape(-1)
        self.A = A
        return self

    def update_embedding(self, dim):
        N = self.A.shape[0]
        Z = sparse.csr_matrix((N, N))
        I = sparse.identity(N, format="csr")
        D = sparse.diags(self.deg)
        B = sparse.bmat([[Z, D - I], [-I, self.A]], format="csr")
        s, v = sparse.linalg.eigs(B, k=dim + 1)
        order = np.argsort(s)
        s, v = s[order], v[:, order]
        s, v = s[1:], v[:, 1:]
        v = v[N:, :]
        c = np.array(np.linalg.norm(v, axis=0)).reshape(-1)
        v = v @ np.diag(1 / c)
        self.in_vec = v
