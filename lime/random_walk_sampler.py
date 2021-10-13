"""This module contains the sentence generator based on random walks.

This process can be a bottleneck of many graph embeddings methods, and thus fast simulation of random walks are critical for large networks.

To this end, we leverage the structure of scipy.csr_matrix. This csr_matrix stores the adjacency matrix in a way that makes it easy to simulate random walks. Specifically, let A is the adjacency matrix in form of scipy.csr_matrix. Then, A has the following perivate members: A.data, A.indptr, and A.indices.

Now, consider simulating a random walk in node i in a network, and finding the next node j to move.
The neighbors of node i can be obtained by
````python
- A.indices[A.indptr[i]:A.indptr[i+1]]
````
connected by edges with weights
````python
- A.data[A.indptr[i]:A.indptr[i+1]].
````
Finding the next node to move is, therefore, sampling from `A.indices[A.indptr[i]:A.indptr[i+1]]` with weight proportional to `A.data[A.indptr[i]:A.indptr[i+1]]`.
"""
import numba
import numpy as np
from scipy import sparse

from lime import utils


class RWSentenceGenerator:
    def __init__(self, A, num_walks, walk_length, restart_prob, chunk_size=100000):
        self.num_walks = num_walks
        self.restart_prob = restart_prob
        self.walk_length = walk_length
        self.num_nodes = A.shape[0]
        self.node_orders = np.repeat(np.arange(self.num_nodes), self.num_walks)
        np.random.shuffle(self.node_orders)

        self.chunk_it = 0
        self.chunk_size = chunk_size
        self.it = 0
        self.maxIt = self.num_nodes * self.num_walks

        self.P = self._calc_cum_trans_prob(A)
        self.restart_prob = 0
        self.walks = []

    def __iter__(self):
        return self

    def __next__(self):
        if self.it == self.maxIt:
            np.random.shuffle(self.node_orders)
            self.it = 0
            raise StopIteration
        result = self._generate_sentence()
        self.it += 1
        return result

    def _generate_sentence(self):
        if self.chunk_it == len(self.walks):
            start_node_ids = self.node_orders[
                self.it : np.minimum(self.it + self.chunk_size, len(self.node_orders))
            ]
            self.walks = _simulate_simple_walk(
                A_indptr=self.P.indptr,
                A_indices=self.P.indices,
                A_data=self.P.data.astype(float),
                num_nodes=self.P.shape[0],
                restart_prob=float(self.restart_prob),
                start_node_ids=start_node_ids.astype(int),
                walk_length=self.walk_length,
                restart_at_dangling=False,
                random_teleport=False,
            )
            self.walks = self.walks.astype(str).tolist()
            self.chunk_it = 0
        result = self.walks[self.chunk_it]
        self.chunk_it += 1
        return result

    #
    # Helper function
    #
    def _calc_cum_trans_prob(self, A):
        P = A.copy()
        for i in range(A.shape[0]):
            outdeg = np.sum(P.data[P.indptr[i] : P.indptr[i + 1]])
            P.data[P.indptr[i] : P.indptr[i + 1]] = np.cumsum(
                P.data[P.indptr[i] : P.indptr[i + 1]]
            ) / np.maximum(outdeg, 1e-12)
        return P


@numba.jit(nopython=True, cache=True, parallel=True)
def _simulate_simple_walk(
    A_indptr,
    A_indices,
    A_data,  # should be cumulative
    num_nodes,
    restart_prob,
    start_node_ids,
    walk_length,
    restart_at_dangling,
    random_teleport,
):
    """Random walk simulator.

    :param window_length: window length, defaults to 10
    :type window_length: int, optional
    :param A_indptr: A private member of scipy.sparse.csr_matrix
    :type A_indptr: csr_matrix.indptr
    :param A_indices: A private member of scipy.sparse.csr_matrix
    :type A_indices: csr_matrix.indices
    :param A_data: A private member of scipy.sparse.csr_matrix
    :type A_data: csr_matrix.data
    :param num_nodes: number of nodes in the network
    :type num_nodes: int
    :param restart_prob: Probability of teleporting back to the starting node.
    :type restart_prob: float
    :param start_node_ids: array of starting nodes
    :type start_node_ids: numpy.ndarray
    :param random_teleport: Probability of teleporting to a node chosen randomly from the network.
    :type random_teleport: float
    :return: walks: List of the trajectories of random walks
    :rtype: walks: list of list
    """
    # Alocate a memory for recording a walk
    walks = -np.ones((len(start_node_ids), walk_length), dtype=np.int32)
    for sample_id in range(len(start_node_ids)):
        start_node = start_node_ids[sample_id]
        # Record the starting node
        visit = start_node
        walks[sample_id, 0] = visit
        for t in range(1, walk_length):
            # Compute the number of neighbors
            outdeg = A_indptr[visit + 1] - A_indptr[visit]
            # If reaches to an absorbing state, finish the walk
            # the random walker is teleported back to the starting node
            # or Random walk with restart
            if outdeg == 0:
                if restart_at_dangling:
                    if random_teleport:
                        next_node = np.random.randint(0, num_nodes)
                    else:
                        next_node = start_node
                else:
                    if t == 1:  # when starting from sink
                        pass
                        walks[sample_id, t] = visit
                    break
            elif np.random.rand() <= restart_prob:
                if random_teleport:
                    next_node = np.random.randint(0, num_nodes)
                else:
                    next_node = start_node
            else:
                # find a neighbor by a roulette selection
                _next_node = np.searchsorted(
                    A_data[A_indptr[visit] : A_indptr[visit + 1]],
                    np.random.rand(),
                    side="right",
                )
                next_node = A_indices[A_indptr[visit] + _next_node]
            # Record the transition
            walks[sample_id, t] = next_node
            # Move
            visit = next_node
    return walks
