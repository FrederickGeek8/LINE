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
    output = [0] * matrix.shape[1]
    count = 0
    for idx in unique[1]:
        output[idx] = count
        count += 1

    u = set(unique[1])
    v = set(range(matrix.shape[1]))
    for i in v.difference(u):
        for j in matrix:
            if np.array_equal(matrix[i], matrix[j]):
                output[i] = output[j]

    return output

def generate_dist(graph, power=0.75):
    # neg_avg = 0
    nodes_prob = np.array(graph.out_degree(weight='weight'))[:, 1]
    # nodes_prob = np.zeros((graph.number_of_nodes()))
    # for start, end, weight in graph.edges(data='weight'):
        # nodes_prob[start] += weight
    
    nodes_prob = np.power(nodes_prob, power)
    nodes_prob = nodes_prob / np.sum(nodes_prob)
    # print(nodes_prob)

    return nodes_prob