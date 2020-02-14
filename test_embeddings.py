import graph_creator
import networkx as nx
import matplotlib.pyplot as plt
import tokenize_files
import numpy as np
from numpy.linalg import norm
from heapq import nlargest
from line_algo import line
from argparse import Namespace


file_paths = graph_creator.get_all_paths('C:/Users/Theofilos/Desktop/Test')

# Create the graph of the training set
libraries, keywords, training_graph = graph_creator.create_graph(file_paths)

# Store graph in a file
nx.write_gpickle(training_graph, "line_algo/data/lib_rec.gpickle")

args = Namespace(embedding_dim=2, batch_size=2, K=1, proximity="first-order", learning_rate=0.025,
                     mode="train", num_batches=1000, total_graph=True, graph_file="line_algo/data/lib_rec.gpickle")
    # Dictionary with the embedding of every library using L.I.N.E. algorithm
embeddings = line.main(args)

graph_creator.plot_graph(libraries,keywords,training_graph)

emb_x = [arr.tolist()[0] for arr in embeddings.values()]
emb_y = [arr.tolist()[1] for arr in embeddings.values()]
plt.figure(2)
plt.scatter(emb_x, emb_y)
plt.show()
