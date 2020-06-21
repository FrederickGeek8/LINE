from line_module import LINE
from sklearn.metrics.pairwise import cosine_similarity
from math import log
import torch
import torch.optim as optim
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sampling import AliasMethod, node_sample
from utils import common_neighbors, context_encoding, generate_dist

if __name__ == "__main__":
    block_size = 10
    order = 1
    
    matrix = np.block([
        [10 * np.ones((block_size,block_size)), np.zeros((block_size,block_size))],
        [np.zeros((block_size, block_size)), 10 * np.ones((block_size, block_size))]
    ])

    np.fill_diagonal(matrix, 0)
    graph = nx.from_numpy_matrix(matrix, create_using=nx.DiGraph())

    edges = torch.LongTensor(list(graph.edges(data='weight')))
    # print(graph.number_of_nodes())

    v_i = edges[:, 0]
    v_j = edges[:, 1]

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


    # Negative sampling
    power = 0.75
    nodes_prob = generate_dist(graph)
    negative_sampling = AliasMethod(np.arange(0, graph.number_of_nodes()), nodes_prob)

    for epoch in range(100):
        print(f"Epoch {epoch}")
        
        negative_nodes = torch.LongTensor(node_sample(v_i, edges[:, 1], negative_sampling))

        line.zero_grad()
        loss = line(v_i, v_j, negative_nodes)
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