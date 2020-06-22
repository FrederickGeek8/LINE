from line_module import LINE
from sklearn.metrics.pairwise import cosine_similarity
from math import log
from torch.utils.tensorboard import SummaryWriter
from sampling import AliasMethod, node_sample
from utils import common_neighbors, context_encoding, generate_dist
import torch
import argparse
import torch.optim as optim
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

if __name__ == "__main__":
    power = 0.75
    parser = argparse.ArgumentParser()
    parser.add_argument("-block", "--block_size", type=int, default=100)
    parser.add_argument("-order", "--order", type=int, default=1)
    parser.add_argument("-batch", "--batch_size", type=int, default=5)
    parser.add_argument("-epochs", "--epochs", type=int, default=10)
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.025)
    parser.add_argument("-tb",
                        "--tensorboard_path",
                        type=str,
                        default="./runs/")
    args = parser.parse_args()

    matrix = np.block([[
        1 * np.ones((args.block_size, args.block_size)),
        np.zeros((args.block_size, args.block_size))
    ],
    [
        np.zeros((args.block_size, args.block_size)),
        1 * np.ones((args.block_size, args.block_size))
    ]])

    np.fill_diagonal(matrix, 0)
    graph = nx.from_numpy_matrix(matrix, create_using=nx.DiGraph())

    if args.order == 2:
        # print(edges.shape[0])
        tmp = []
        context = common_neighbors(graph)
        uni = np.array(context_encoding(context))

    model = LINE(graph.number_of_nodes(), latent_dim=2, order=args.order)

    if torch.cuda.is_available():
        model = model.cuda()

    opt = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)

    # Negative sampling
    nodes_prob, edges_prob = generate_dist(graph)
    negative_sampling = AliasMethod(np.arange(0, graph.number_of_nodes()),
                                    nodes_prob)
    edge_selector = AliasMethod(edges_prob[:, :2], edges_prob[:, 2])
    batch_range = edges_prob.shape[0] // args.batch_size

    writer = SummaryWriter(args.tensorboard_path)

    nodes = torch.LongTensor(list(graph.nodes()))

    for epoch in range(args.epochs):
        print(f"Epoch {epoch}")
        for b in range(batch_range):
            edges = torch.LongTensor(edge_selector.sample(args.batch_size))
            v_i = edges[:, 0]
            v_j = edges[:, 1]

            if args.order == 2:
                fin = uni[v_j]
                v_j = torch.LongTensor(fin)

            negative_nodes = torch.LongTensor(
                node_sample(v_i, v_j, negative_sampling))
            
            if torch.cuda.is_available():
                v_i = v_i.cuda()
                v_j = v_j.cuda()
                negative_nodes = negative_nodes.cuda()

            model.zero_grad()
            loss = model(v_i, v_j, negative_nodes)
            loss.backward()

            # print(list(model.parameters())[1])

            opt.step()

            if b % (batch_range // 5) == 0:
                print("Loss:", loss.item())
                embedding = model.embedding(nodes).detach().numpy()
                figure = plt.figure()
                # print(embedding.shape)

                plt.scatter(embedding[:, 0], embedding[:, 1], s=14)

                writer.add_figure('images', figure, epoch * batch_range + b)
                writer.add_scalar('loss', loss, epoch * batch_range + b)

    writer.close()

    # print(graph.nodes)
    # exit(0)

    embedding = model.embedding(nodes).detach().numpy()
    # print(embedding.shape)

    plt.scatter(embedding[:, 0], embedding[:, 1], s=14)

    plt.show()