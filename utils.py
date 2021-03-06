import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def common_neighbors(graph):
    # Node i (row) is connected to j (col)
    adjacency_matrix = nx.to_numpy_matrix(graph)
    pairwise = cosine_similarity(adjacency_matrix)
    return pairwise


def context_encoding(matrix):
    unique = np.unique(matrix, return_index=True, axis=0)
    # print(unique)
    # exit(0)
    output = [0] * matrix.shape[1]
    count = 0
    for idx in unique[1]:
        output[idx] = count
        count += 1

    u = set(unique[1])
    v = set(range(matrix.shape[1]))
    for i in v.difference(u):
        for j in range(matrix.shape[1]):
            if np.array_equal(matrix[i], matrix[j]):
                output[i] = output[j]

    return output

def generate_dist(graph, power=0.75):
    nodes_prob = np.array(graph.out_degree(weight='weight'))[:, 1]
    nodes_prob = np.power(nodes_prob, power)
    nodes_prob = nodes_prob / np.sum(nodes_prob)

    edges = np.array(list(graph.edges(data='weight', default=1.0)))
    edges[:, 2] = edges[:, 2] / np.sum(edges[:, 2])

    # print(edges)

    return nodes_prob, edges