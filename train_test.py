from line_module import LINE
from sklearn.metrics.pairwise import cosine_similarity
from math import log
import torch
import torch.optim as optim
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

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

if __name__ == "__main__":
    block_size = 10
    order = 2
    
    matrix = np.block([
        [1 * np.ones((block_size,block_size)), np.zeros((block_size,block_size))],
        [np.zeros((block_size, block_size)), 1 * np.ones((block_size, block_size))]
    ])

    np.fill_diagonal(matrix, 0)
    graph = nx.from_numpy_matrix(matrix, create_using=nx.DiGraph())

    edges = torch.LongTensor(list(graph.edges(data='weight')))
    # print(graph.number_of_nodes())

    v_i = edges[:, 0]
    v_j = edges[:, 1]
    w_ij = edges[:, 2]

    if order == 2:
        # print(edges.shape[0])
        tmp = []
        context = common_neighbors(graph)
        uni = np.array(context_encoding(context))
        fin = uni[v_j]
        # print(fin)
        v_j = torch.LongTensor(fin)

    line = LINE(graph.number_of_nodes(), latent_dim=2, order=order)

    opt = optim.SGD(line.parameters(), lr=0.25, momentum=0.9)

    for epoch in range(100):
        print(f"Epoch {epoch}")
        
        line.zero_grad()
        loss = line(v_i, v_j, w_ij)
        loss.backward()

        # print(list(line.parameters())[1])
        
        opt.step()
        print(loss.item())
    
    nodes = torch.LongTensor(list(graph.nodes()))

    print(graph.nodes)
    # exit(0)

    embedding = line.embedding(nodes).detach().numpy()
    print(embedding.shape)

    plt.scatter(embedding[:,0], embedding[:,1], s=14)
    
    plt.show()