import random
import os
import graph_creator
import networkx as nx
import matplotlib.pyplot as plt
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
import statistics
from sklearn.decomposition import PCA
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler
from node2vec import Node2Vec
from gensim.models import Word2Vec
from graph_line_by_line import line_by_line_graph

# Calculate the hit rate of top predicted libraries
def calculate_hit_rate(actual_libraries, top_libraries_predicted):
    predicted_count = 0
    for library_predicted in top_libraries_predicted:
        if library_predicted in actual_libraries:
            predicted_count = predicted_count + 1
    hit_rate = predicted_count / len(top_libraries_predicted)
    return hit_rate

#Calculate the similarity for each library with every keyword
def calculate_similarity(libraries, keywords,embeddings, path_keywords, model="", method="dot" ):
    # Initialize a dictionary with 0 for the cos similarity of every library
    sim = {library: 0 for library in libraries}
    idf_dict= tokenize_files.tf_idf(file_paths)
    if method == "cosine":

        for library in libraries:
            for keyword in path_keywords:
                if keyword in keywords:
                    lib_array = embeddings[library]
                    keyword_array = embeddings[keyword]
                    cos_similarity = np.dot(lib_array, keyword_array) / (norm(lib_array) * norm(keyword_array))
                    if keyword in idf_dict.keys():
                        sim[library] = sim[library] + cos_similarity*idf_dict[keyword]
                    else:
                        sim[library] = sim[library] + cos_similarity
    elif method=="dot":
        for library in libraries:
            for keyword in path_keywords:
                if keyword in keywords:
                    lib_array = embeddings[library]
                    keyword_array = embeddings[keyword]
                    dot_similarity = np.dot(lib_array, keyword_array)
                    if keyword in idf_dict.keys():
                        sim[library] = sim[library] + dot_similarity*idf_dict[keyword]
                    else:
                        sim[library] = sim[library] + dot_similarity
    else:
        for library in libraries:
            for keyword in path_keywords:
                if keyword in keywords:
                    lib_features = scaler.transform([embeddings[library]])
                    key_features=  scaler.transform([embeddings[keyword]])
                    function_similarity = model.predict(np.multiply(lib_features,key_features))
                    if keyword in idf_dict.keys():
                        sim[library] = sim[library] + int(function_similarity[0])*idf_dict[keyword]
                    else:
                        sim[library] = sim[library] + int(function_similarity[0])
    return sim


# Predict the libraries in this file using the adjacency matrix and an graph signal
# with 1s where keyword exists and 0 where keyword doesn't
def predict_libraries(path_keywords, adj_matrix, nodes_names):

    node_atr = list(repeat(0, len(adj_matrix)))
    for keyword in path_keywords:
        if keyword in nodes_names:
            node_atr[nodes_names.index(keyword)] = 1
    graph_filter = np.matmul(adj_matrix, np.asarray(node_atr))
    value_list = graph_filter.tolist()[0]
    return value_list

# Create a training set for the similarity prediction model
# Training features: products of each pair of feature's value
# Training values:  1 when edge exists and 0 when not
# Almost equal number of positive and negative samples
def create_training_set(training_graph, embeddings, libraries, keywords):

    training_features = []
    training_values = []
    for node1, node2, weight in training_graph.edges(data=True):
        training_features.append(np.multiply(embeddings[node1], embeddings[node2]))
        training_values.append(1)

    negative_values_per_node = 480
    nodes_names = list(training_graph.nodes())
    random.seed(20)
    for node in nodes_names:
        count = 0
        if node in libraries:
            while count < negative_values_per_node:
                random_node = random.choice(nodes_names)
                if random_node in keywords and not (training_graph.has_edge(node, random_node)):
                    training_features.append(np.multiply(embeddings[node], embeddings[random_node]))
                    training_values.append(0)
                    count = count + 1
    print(Counter(training_values))

    return training_features, training_values

# Train a ML model that predicts if there is a an edge between two nodes
# Return the model and the scaler that was used
def train_relation_model(training_features, training_values):

    scaler = StandardScaler()
    scaler.fit(training_features)
    scaler.transform(training_features)

    # model= LinearSVC(random_state=0, tol=1e-5)
    model = LogisticRegression(random_state=0, penalty="l2")
    train_f = np.matrix(training_features)
    model.fit(train_f, training_values)
    print("Accuracy of Relation model: ", model.score(train_f, training_values))

    return scaler, model

# Return the embeddings of the graph using the L.I.N.E. or teh node
def get_embeddings(training_graph, embedddings_method="line",proximity_method="both"):

    if embedddings_method=="line":
        if proximity_method == "both":
            args = Namespace(embedding_dim=16, batch_size=16, K=5, proximity="first-order", learning_rate=0.025,
                             mode="train", num_batches=1000, total_graph=True,
                             graph_file="line_algo/data/lib_rec.gpickle")
            # Dictionary with the embedding of every library using L.I.N.E. algorithm
            embeddings_first = line.main(args)

            args = Namespace(embedding_dim=16, batch_size=16, K=5, proximity="second-order", learning_rate=0.025,
                             mode="train", num_batches=1000, total_graph=True,
                             graph_file="line_algo/data/lib_rec.gpickle")
            embeddings_second = line.main(args)
            embeddings = {}
            for node in training_graph.nodes():
                embeddings[node] = np.concatenate((embeddings_first[node], 0.3 * embeddings_second[node]), axis=None)
        else:
            args = Namespace(embedding_dim=16, batch_size=16, K=5, proximity="first-order", learning_rate=0.025,
                             mode="train", num_batches=1000, total_graph=True,
                             graph_file="line_algo/data/lib_rec.gpickle")
            # Dictionary with the embedding of every library using L.I.N.E. algorithm
            embeddings = line.main(args)
    elif embedddings_method=="node2vec":
        # Precompute probabilities and generate walks - **ON WINDOWS ONLY WORKS WITH workers=1**
        node2vec = Node2Vec(training_graph, dimensions=16, walk_length=80, num_walks=50,
                            workers=1, temp_folder="./temp_graph",p=2, q=0.5)  # Use temp_folder for big graphs
        # Embed nodes
        model = node2vec.fit(window=4, min_count=1,batch_words=4)

        embeddings={node:model.wv[node] for node in training_graph.nodes}
        model.save('./embeddings.model')

    return embeddings

if __name__ == "__main__":
    # Hyper parameters
    # Graph-method: co-occur, line-by-line
    # have_embeddings: True, False
    # random_predict: True, False
    graph_method = "co-occur"
    have_embeddings = True
    random_predict = False

    # Get all the paths inside the directory path provided before
    file_paths = graph_creator.get_all_paths('keras\\tests')

    # Training 80% Testing 20%
    training_set, test_set = train_test_split(file_paths, test_size = 0.2, random_state = 40)

    # Create the graph of the training set with the appropriate method
    if graph_method=="co-occur":
        libraries, keywords, training_graph = graph_creator.create_graph(training_set)
    elif graph_method=="line-by-line":
        libraries, keywords, training_graph = line_by_line_graph(training_set)

    #Store graph in a file
    nx.write_gpickle(training_graph,"line_algo/data/lib_rec.gpickle")

    if not have_embeddings:
        hit_rate=[]
        auc=[]
        ndcg=[]
        for file_path in test_set:
            adj_matrix = nx.to_numpy_matrix(training_graph)
            print('Predict path: ', file_path)

            # Get the path's libraries and keyword for the specific file in the path
            path_libraries, path_keywords = tokenize_files.get_libs_and_keywords(file_path)
            print("Libraries in this file: ", len(path_libraries))
            nodes_names= list(training_graph.nodes())
            if random_predict:
                sim ={library:random.randint(1,1000) for library in libraries}
                predicted_libraries = nlargest(5, sim, key=sim.get)
            else:
                # Predict libraries in this path
                lib_predicted_values = predict_libraries(path_keywords, adj_matrix, nodes_names)

                sim = {}
                for lib_value in lib_predicted_values:
                    if nodes_names[lib_predicted_values.index(lib_value)] in libraries:
                        sim[nodes_names[lib_predicted_values.index(lib_value)]] = lib_value

                predicted_libraries = nlargest(20, sim, key=sim.get)

            # Get the largest 5 values
            print("Libraries predicted: ", predicted_libraries)
            print("Path libraries:", path_libraries, "\n")

            # Hit rate for Top-5 libraries
            hit_rate_temp = calculate_hit_rate(path_libraries, predicted_libraries)
            hit_rate.append(hit_rate_temp)
            print("Hit Rate @", len(predicted_libraries), ": ", hit_rate_temp)

            # Calculate AUC
            labels = [1 if library in path_libraries else 0 for library in sim.keys()]
            conf = list(sim.values())
            auc.append(roc_auc_score(np.array(labels), np.array(conf)))
            print("ROC AUC: ", roc_auc_score(np.array(labels), np.array(conf)), "\n")

            # Calculate Normalized Cumulative Score
            # Relevance score=1 if a library that was predicted is in path's libraries
            ndcg.append(ndcg_score([np.array(labels)], [np.array(conf)]))
            print("Discounted Cumulative Gain: ", ndcg_score([np.array(labels)], [np.array(conf)]), '\n')

    else:
        embeddings_method="node2vec"
        similarity_method = "dot"
        embeddings=get_embeddings(training_graph, embeddings_method, proximity_method="both")
        if similarity_method=='function':
            # Create training set for the model of the similarity prediction
            training_features, training_values = create_training_set(training_graph, embeddings, libraries, keywords)
            # Train the relation model
            scaler, model= train_relation_model(training_features, training_values)

        hit_rate = []
        auc = []
        ndcg = []
        for file_path in test_set:
            # Get the path's libraries and keyword for the specific file in the path
            print('Predict path: ', file_path)
            path_libraries, path_keywords = tokenize_files.get_libs_and_keywords(file_path)
            print("Number of libraries in this file: ", len(path_libraries))

            # Calculate similarity and save it in a dictionary
            if similarity_method=="function":
                sim=calculate_similarity(libraries, keywords,embeddings, path_keywords,model, method=similarity_method)
            else:
                sim = calculate_similarity(libraries, keywords, embeddings, path_keywords, method=similarity_method)

            # Get the largest 5 values
            predicted_libraries = nlargest(10, sim, key = sim.get)
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

    print(" \n Hit rate @",len(predicted_libraries)," \n        Average: ",sum(hit_rate)/len(hit_rate))
    print("        Range: ", min(hit_rate)," - ", max(hit_rate))

    print(" \n AUC" ," \n        Average: ", sum(auc) / len(auc))
    print("        Range: ", min(auc), " - ", max(auc))

    print(" \n NDCG", " \n        Average: ", sum(ndcg) / len(ndcg))
    print("        Range: ", min(ndcg), " - ", max(ndcg))

