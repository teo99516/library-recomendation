import random
import os
import graph_creator
import networkx as nx
import matplotlib.pyplot as plt
import tokenize_files
from itertools import repeat
import numpy as np
from heapq import nlargest
from sklearn import metrics
import matplotlib.pyplot as plt


# Choose a portion of the data set randomly
def random_split(file_paths, percentage_training):
    random.shuffle(file_paths)
    training_set = file_paths[0:int(percentage_training * len(file_paths))]
    test_set = file_paths[int(percentage_training * len(file_paths)):]

    return training_set, test_set


# Calculate the hit rate of top predicted libraries
def get_hit_rate(actual_libraries, top_libraries_predicted):
    predicted_count = 0
    for library_predicted in top_libraries_predicted:
        if library_predicted in actual_libraries:
            predicted_count = predicted_count + 1

    hit_rate = predicted_count / len(top_libraries_predicted)

    return hit_rate


def get_auc(labels, confidence):
    fpr, tpr, thresholds = metrics.roc_curve(labels, confidence, pos_label=1)
    # plt.plot(fpr, tpr)
    # plt.show()
    auc = metrics.auc(fpr, tpr)
    print("ROC AUC = ", auc, "\n")

    return auc


# Predict the libraries in this file using the adjacency matrix and an graph signal
# with 1s where keyword exists and 0 where keyword doesn't
def predict_libraries(path_keywords, node_atr, adj_matrix):
    for keyword in path_keywords:
        if keyword in nodes_names:
            node_atr[nodes_names.index(keyword)] = 1

    graph_filter = np.matmul(adj_matrix, np.asarray(node_atr))
    value_list = graph_filter.tolist()
    value_list = value_list[0]

    return value_list


# Get libraries with the highest values
def get_highest_predicted_values(confidence, nodes_names, libraries_count):
    print(' Highest ', libraries_count, ' values of libraries predicted: ')
    max_indices = nlargest(libraries_count, range(len(confidence)), confidence.__getitem__)

    highest_values_libraries = []
    for index in max_indices:
        print('  ', nodes_names[index], ' with confidence: ', confidence[index])
        highest_values_libraries.append(nodes_names[index])
    print("\n")

    return highest_values_libraries


if __name__ == "__main__":

    # Get all the paths inside the directory path provided before
    dir_path = os.getcwd() + '\keras\\tests'
    file_paths = graph_creator.get_all_paths(dir_path)

    # Training 80% Testing 20%
    training_set, test_set = random_split(file_paths, 0.8)

    # Create the graph of the training set
    libraries, keywords, training_graph = graph_creator.create_graph(training_set)
    adj_matrix = nx.to_numpy_matrix(training_graph)

    for file_path in test_set:
        print('Predict path: ', file_path)

        # Get the path's libraries and keyword for the specific file in the path
        path_libraries, path_keywords = tokenize_files.get_libs_and_keywords(file_path)
        print("Libraries in this file: ", len(path_libraries))
        nodes_names = list(training_graph.nodes)
        node_atr = list(repeat(0, len(nodes_names)))

        # Predict libraries in this path
        lib_predicted_values = predict_libraries(path_keywords, node_atr, adj_matrix)

        # Create array with 1 for libraries predicted in this path and 0s for libraries that was not
        labels = [1 if node in path_libraries else 0 for node in nodes_names]

        # Compute confidence value of each library
        confidence = [100 * lib_value / sum(lib_predicted_values) for lib_value in lib_predicted_values]

        # Return the 5 libraries with the highest score
        highest_values_libraries = get_highest_predicted_values(confidence, nodes_names, 5)

        # Hit rate for Top-5 libraries
        hit_rate = get_hit_rate(path_libraries, highest_values_libraries)
        print('Hit Rate @5: ', hit_rate)

        # Calculate AUC
        auc = get_auc(np.array(labels), np.array(confidence))