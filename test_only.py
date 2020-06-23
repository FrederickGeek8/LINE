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
    parser.add_argument("-graph", "--graph_path", type=str, required=True)
    parser.add_argument("-order", "--order", type=int, default=1)
    parser.add_argument("-load", "--load_path", type=str, required=True)
    args = parser.parse_args()

    graph = nx.read_edgelist(args.graph_path, nodetype=int)
    graph = nx.relabel.convert_node_labels_to_integers(graph)
    graph = graph.to_directed()
    
    nodes = torch.LongTensor(list(graph.nodes()))
    model = LINE(graph.number_of_nodes(), latent_dim=2, order=args.order)
    model_pt = torch.load(f"{args.load_path}")
    model.load_state_dict(model_pt)
    model.eval()
    
    if torch.cuda.is_available():
        model = model.cuda()
        nodes = nodes.cuda()
    
    embedding = model.embedding(nodes).cpu().detach().numpy()
    # print(embedding.shape)

    plt.scatter(embedding[:, 0], embedding[:, 1], s=14)

    plt.show()

    