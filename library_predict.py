import random
import os
import graph_creator
import networkx as nx
import matplotlib.pyplot as plt
import tokenize_files
import numpy as np
from numpy import dot
from numpy.linalg import norm
from itertools import repeat
from heapq import nlargest
from sklearn import metrics
from line_algo import line
from sklearn.metrics.pairwise import cosine_similarity
from argparse import Namespace
from sklearn.preprocessing import MinMaxScaler

# Choose a portion of the data set randomly
def random_split(file_paths, percentage_training=0.8, is_random=True):
    if is_random:
        random.shuffle(file_paths)
    training_set = file_paths[0:int(percentage_training * len(file_paths))]
    test_set = file_paths[int(percentage_training * len(file_paths)):]
    return training_set, test_set

# Calculate the hit rate of top predicted libraries
def calculate_hit_rate(actual_libraries, top_libraries_predicted):
    predicted_count = 0
    for library_predicted in top_libraries_predicted:
        if library_predicted in actual_libraries:
            predicted_count = predicted_count + 1
    hit_rate = predicted_count / len(top_libraries_predicted)
    return hit_rate


def calculate_auc(labels, confidence):
    fpr, tpr, thresholds = metrics.roc_curve(labels, confidence, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    # plt.plot(fpr, tpr)
    # plt.show()
    print("Thresholds: ",thresholds)
    print("ROC AUC = ", auc, "\n")
    return auc

# Predict the libraries in this file using the adjacency matrix and an graph signal
# with 1s where keyword exists and 0 where keyword doesn't
def predict_libraries(path_keywords, adj_matrix):

    node_atr = list(repeat(0, len(adj_matrix)))
    for keyword in path_keywords:
        if keyword in nodes_names:
            node_atr[nodes_names.index(keyword)] = 1
    graph_filter = np.matmul(adj_matrix, np.asarray(node_atr))
    value_list = graph_filter.tolist()[0]
    return value_list


# Get libraries with the highest values
def highest_predicted_values(confidence, nodes_names, libraries_count):
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
    file_paths = graph_creator.get_all_paths('keras\\tests')

    # Training 80% Testing 20%
    training_set, test_set = random_split(file_paths, 0.8, is_random=False)

    # Create the graph of the training set
    libraries, keywords, training_graph = graph_creator.create_graph(training_set)
    #Store graph in a file
    nx.write_gpickle(training_graph,"line_algo/data/lib_rec.gpickle")

    have_embeddings= True
    # Cosine or dot-similarity
    similarity= "dot"
    if not have_embeddings:
        hit_rate=[]
        for file_path in test_set:
            adj_matrix = nx.to_numpy_matrix(training_graph)
            print('Predict path: ', file_path)

            # Get the path's libraries and keyword for the specific file in the path
            path_libraries, path_keywords = tokenize_files.get_libs_and_keywords(file_path)
            print("Libraries in this file: ", len(path_libraries))
            nodes_names = list(training_graph.nodes)

            # Predict libraries in this path
            lib_predicted_values = predict_libraries(path_keywords,  adj_matrix)

            # Compute confidence value of each library
            confidence = [100 * lib_value / sum(lib_predicted_values) for lib_value in lib_predicted_values]

            # Return the 5 libraries with the highest score
            predicted_libraries = highest_predicted_values(confidence, nodes_names, 5)

            # Create array with 1 for libraries predicted in this path and 0s for libraries that was not
            labels = [1 if node in predicted_libraries else 0 for node in nodes_names]

            # Hit rate for Top-5 libraries
            hit_rate_temp = calculate_hit_rate(path_libraries, predicted_libraries)
            hit_rate.append(hit_rate_temp)
            print('Hit Rate @5: ', hit_rate_temp, "\n")

            # Calculate AUC
            auc = calculate_auc(np.array(labels), np.array(confidence))
    else:
        args = Namespace(embedding_dim=128, batch_size=128, K=5, proximity="first-order", learning_rate=0.025,
                         mode="train", num_batches=1000, total_graph=True, graph_file="line_algo/data/lib_rec.gpickle")
        # Dictionary with the embedding of every library
        embeddings = line.main(args)
        hit_rate=[]
        for file_path in test_set:
            # Get the path's libraries and keyword for the specific file in the path
            print('Predict path: ', file_path)
            path_libraries, path_keywords = tokenize_files.get_libs_and_keywords(file_path)
            print("Libraries in this file: ", len(path_libraries))

            if similarity == "cosine":
                # Initialize a dictionary with 0 for the cos similarity of every library
                sim={library:0 for library in libraries}

                for library in libraries:
                    for keyword in path_keywords:
                       if keyword in keywords:
                            lib_array=embeddings[library]
                            keyword_array=embeddings[keyword]
                            cos_similarity = dot(lib_array, keyword_array) / (norm(lib_array) * norm(keyword_array))
                            sim[library] = sim[library] + cos_similarity

                predicted_libraries = nlargest(5, sim, key = sim.get)
                print(predicted_libraries)
                predicted_values = [sim[value] for value in predicted_libraries]
            elif similarity=="dot":
                # Initialize a dictionary with 0 for the cos similarity of every library
                sim = {library: 0 for library in libraries}

                for library in libraries:
                    for keyword in path_keywords:
                        if keyword in keywords:
                            lib_array = embeddings[library]
                            keyword_array = embeddings[keyword]
                            dot_similarity = dot(lib_array, keyword_array)
                            sim[library] = sim[library] + dot_similarity

                predicted_libraries = nlargest(5, sim, key=sim.get)
                print(predicted_libraries)
                predicted_values = [sim[value] for value in predicted_libraries]

            # Hit rate for Top-5 libraries
            hit_rate_temp=calculate_hit_rate(path_libraries, predicted_libraries)
            hit_rate.append(hit_rate_temp)
            print('Hit Rate @5: ', hit_rate_temp, "\n")

            # Calculate AUC
            labels = [1 if library in predicted_libraries else 0 for library in  sim.keys()]
            scaler = MinMaxScaler(feature_range=(0, 1))
            conf = [ [value] for value in sim.values() ]
            scaler.fit(conf)
            confidence= scaler.transform(conf)
            conf = [value[0] for value in confidence]
            auc = calculate_auc(np.array(labels), np.array(conf))

    print(" \n Average hit rate: ", sum(hit_rate)/len(hit_rate))
    print(" Hit rate range: ", min(hit_rate)," - ", max(hit_rate))
    print(" Standard Deviation: ", np.std(np.array(hit_rate)) )

    #print(len(training_graph.edges(['random'])))
    #print(len(training_graph.edges(['time'])))
    #print(len(training_graph.edges(['__future__.print_function'])))


