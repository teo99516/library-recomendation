import networkx as nx
import matplotlib.pyplot as plt
import pickle
from scipy import spatial

if __name__ == "__main__":

    G=nx.read_gpickle('data\lib_rec.gpickle')

    #nx.draw(G)

    #plt.show()

    #Returns a dictionary with the embeddings
    embeddings=nx.read_gpickle('data\embedding_second-order.gpickle')
    print(len(embeddings))

    for key in embeddings:
        result = 1 - spatial.distance.cosine(embeddings[key],)

    #for key in embeddings:
     #   print(embeddings[key])












