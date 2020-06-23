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
    parser.add_argument("-graph", "--graph_path", type=str)
    parser.add_argument("-block", "--block_size", type=int, default=100)
    parser.add_argument("-order", "--order", type=int, default=1)
    parser.add_argument("-batch", "--batch_size", type=int, default=5)
    parser.add_argument("-epochs", "--epochs", type=int, default=10)
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.025)
    parser.add_argument("-sam", "--node_sample_size", type=int, default=5)
    parser.add_argument("-tb", "--tensorboard_path", type=str, default=None)
    args = parser.parse_args()

    print(f"Using order {args.order}")

    graph = nx.read_edgelist(args.graph_path, nodetype=int)
    graph = nx.relabel.convert_node_labels_to_integers(graph)
    graph = graph.to_directed()
    
    model = LINE(graph.number_of_nodes(), latent_dim=2, order=args.order)

    if torch.cuda.is_available():
        model = model.cuda()

    opt = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)

    # Negative sampling
    nodes_prob, edges_prob = generate_dist(graph)
    negative_sampling = AliasMethod(np.arange(0, graph.number_of_nodes()),
                                    nodes_prob)
    edge_selector = AliasMethod(edges_prob[:, :2].astype(int), edges_prob[:, 2])

    batch_range = edges_prob.shape[0] // args.batch_size

    writer = SummaryWriter(args.tensorboard_path)

    nodes = torch.LongTensor(list(graph.nodes()))

    for epoch in range(args.epochs):
        print(f"Epoch {epoch}")
        for b in range(batch_range):
            edges = torch.LongTensor(edge_selector.sample(args.batch_size))
            v_i = edges[:, 0]
            v_j = edges[:, 1]
            
            negative_nodes = torch.LongTensor(
                node_sample(v_i,
                            v_j,
                            negative_sampling,
                            size=args.node_sample_size))

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
                print(
                    f"Done {(epoch * batch_range + b) / (args.epochs * batch_range)}"
                )
                embedding = model.embedding(nodes).cpu().detach().numpy()
                figure = plt.figure()
                # print(embedding.shape)

                plt.scatter(embedding[:, 0], embedding[:, 1], s=14)

                writer.add_figure('images', figure, epoch * batch_range + b)
                writer.add_scalar('loss', loss, epoch * batch_range + b)

    writer.close()

    # print(graph.nodes)
    # exit(0)

    embedding = model.embedding(nodes).cpu().detach().numpy()
    # print(embedding.shape)

    plt.scatter(embedding[:, 0], embedding[:, 1], s=14)

    plt.show()