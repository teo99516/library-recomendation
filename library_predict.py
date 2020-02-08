import random
import os
import graph_creator
import networkx as nx
#import matplotlib.pyplot as plt
import tokenize_files
import numpy as np
from numpy.linalg import norm
from itertools import repeat
from heapq import nlargest
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import ndcg_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from line_algo import line
from argparse import Namespace
from collections import Counter
from sklearn.feature_selection import RFE

# Calculate the hit rate of top predicted libraries
def calculate_hit_rate(actual_libraries, top_libraries_predicted):
    predicted_count = 0
    for library_predicted in top_libraries_predicted:
        if library_predicted in actual_libraries:
            predicted_count = predicted_count + 1
    hit_rate = predicted_count / len(top_libraries_predicted)
    return hit_rate

#Calculate the similarity for each library with every keyword
def calculate_similarity(libraries, keywords,embeddings, path_keywords, model, method="dot" ):
    # Initialize a dictionary with 0 for the cos similarity of every library
    sim = {library: 0 for library in libraries}
    if method == "cosine":
        for library in libraries:
            for keyword in path_keywords:
                if keyword in keywords:
                    lib_array = embeddings[library]
                    keyword_array = embeddings[keyword]
                    cos_similarity = np.dot(lib_array, keyword_array) / (norm(lib_array) * norm(keyword_array))
                    sim[library] = sim[library] + cos_similarity
    elif method=="dot":
        for library in libraries:
            for keyword in path_keywords:
                if keyword in keywords:
                    lib_array = embeddings[library]
                    keyword_array = embeddings[keyword]
                    dot_similarity = np.dot(lib_array, keyword_array)
                    sim[library] = sim[library] + dot_similarity
    else:
        for library in libraries:
            for keyword in path_keywords:
                if keyword in keywords:
                    lib_features = embeddings[library]
                    key_features=  embeddings[keyword]
                    function_similarity = model.predict([np.multiply(lib_features, key_features)])
                    sim[library] = sim[library] + int(function_similarity[0])
    #print(sim)
    return sim


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
    training_set, test_set = train_test_split(file_paths, test_size = 0.2, random_state = 40)

    # Create the graph of the training set
    libraries, keywords, training_graph = graph_creator.create_graph(training_set)
    #Store graph in a file
    nx.write_gpickle(training_graph,"line_algo/data/lib_rec.gpickle")

    have_embeddings= True
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
        args = Namespace(embedding_dim=16, batch_size=16, K=5, proximity="first-order", learning_rate=0.025,
                         mode="train", num_batches=1000, total_graph=True, graph_file="line_algo/data/lib_rec.gpickle")
        # Dictionary with the embedding of every library using L.I.N.E. algorithm
        embeddings = line.main(args)
        hit_rate=[]
        auc=[]
        ndcg=[]
        training_features=[]
        training_values=[]

        for node1, node2, weight in training_graph.edges(data=True):
            training_features.append(np.multiply(embeddings[node1], embeddings[node2]))
            training_values.append(1)

        negative_values_per_node=18
        nodes_names=list(training_graph.nodes())
        random.seed(20)
        for node in nodes_names:
            count=0
            while count<negative_values_per_node:
                random_node=random.choice(nodes_names)
                if not(training_graph.has_edge(node,random_node)):
                    training_features.append(np.multiply(embeddings[node], embeddings[random_node]))
                    training_values.append(0)
                    count=count+1

        #training_features=np.matrix(training_features)[:,[1,2,3,4,5,6,7,8,11,12,14,15]]

        #print(training_values)

        print(Counter(training_values))
        #model= LinearSVC(random_state=0, tol=1e-5)
        model=LogisticRegression(random_state=0, multi_class="ovr")
        #selector= RFE(model, 12, step=1)
        #selector=selector.fit(training_features,training_values)
        #print(selector.ranking_)
        model.fit(training_features,training_values)
        print(model.score(training_features,training_values))


        for file_path in test_set:
            # Get the path's libraries and keyword for the specific file in the path
            print('Predict path: ', file_path)
            path_libraries, path_keywords = tokenize_files.get_libs_and_keywords(file_path)
            print("Number of libraries in this file: ", len(path_libraries))

            # Calculate similarity and save it in a dictionary
            sim=calculate_similarity(libraries, keywords,embeddings, path_keywords, model, method='cosine')

            # Get the largest 20 values
            predicted_libraries = nlargest(5, sim, key = sim.get)
            print("Libraries predicted: ",predicted_libraries)
            print("Path libraries:",path_libraries, "\n")

            # Hit rate for Top-5 libraries
            hit_rate_temp=calculate_hit_rate(path_libraries, predicted_libraries)
            hit_rate.append(hit_rate_temp)
            print("Hit Rate @",len(predicted_libraries),": ", hit_rate_temp)

            # Calculate AUC
            labels = [1 if library in path_libraries else 0 for library in  sim.keys()]
            conf =list(sim.values())
            auc.append(roc_auc_score(np.array(labels), np.array(conf)))
            print("ROC AUC: ", roc_auc_score(np.array(labels), np.array(conf)),"\n")

            #Calculate Normalized Cumulative Score
            # Relevance score=1 if a library that was predicted is in path's libraries
            ndcg.append(ndcg_score([np.array(labels)], [np.array(conf)]))
            print("Discounted Cumulative Gain: ", ndcg_score([np.array(labels)], [np.array(conf)]),'\n')
            # rel_scores=[1 if library in path_libraries else 0 for  library in sorted(sim, key=sim.get, reverse=True)]
            # print(rel_scores)

    print(" \n Hit rate @",len(predicted_libraries)," \n        Average: ",sum(hit_rate)/len(hit_rate))
    print("        Range: ", min(hit_rate)," - ", max(hit_rate))

    print(" \n AUC" ," \n        Average: ", sum(auc) / len(auc))
    print("        Range: ", min(auc), " - ", max(auc))

    print(" \n NDCG", " \n        Average: ", sum(ndcg) / len(ndcg))
    print("        Range: ", min(ndcg), " - ", max(ndcg))




