import random
import os
import graph_creator
import networkx as nx
import matplotlib.pyplot as plt
import tokenize_files
from itertools import repeat
import numpy as np
from heapq import nlargest


# Calculate the accuracy of predicted libraries
def evaluate(actual_libraries, libraries_predicted):
    predicted_count = 0
    for library_predicted in libraries_predicted:
        if library_predicted in actual_libraries:
            predicted_count = predicted_count + 1

    accuracy = predicted_count / len(actual_libraries)

    return accuracy


if __name__ == "__main__":

    dir_path = os.getcwd() + '\keras\\tests'
    # Get all the paths inside the directory path provided before
    file_paths = graph_creator.get_all_paths(dir_path)

    # Choose a portion of the data set randomly
    # Training 80% Testing 20%
    random.shuffle(file_paths)
    training_set = file_paths[0:int(0.8 * len(file_paths))]
    test_set = file_paths[int(0.8 * len(file_paths)):]

    # Create the graph of the training set
    libraries, keywords, G = graph_creator.create_graph(training_set)
    adj_matrix = nx.to_numpy_matrix(G)
    print(adj_matrix)

    for file_path in test_set:

        print('Predict path: ', file_path)

        # Get the path's libraries and keyword for the specific file in the path
        path_libraries, path_keywords = tokenize_files.get_libs_and_keywords(file_path)

        # Make graph signal with value 1 if keyword exists in this graph and 0 if it doesn't
        nodes_names = list(G.nodes)
        node_atr = list(repeat(0, len(nodes_names)))
        for keyword in path_keywords:
            if keyword in nodes_names:
                node_atr[nodes_names.index(keyword)] = 1

        # Multiply adjacency matrix with the signal created before
        graph_filter = np.matmul(adj_matrix, np.asarray(node_atr))
        value_list = graph_filter.tolist()
        value_list = value_list[0]
        print(' Highest 15 values of libraries predicted: ')
        max_indices = nlargest(15, range(len(value_list)), value_list.__getitem__)
        libraries_predicted = []
        for index in max_indices:
            print('  ', nodes_names[index])
            libraries_predicted.append(nodes_names[index])

        accuracy = evaluate(path_libraries, libraries_predicted)
        print('\n', "Accuracy of libraries predicted: ", accuracy, '\n')
