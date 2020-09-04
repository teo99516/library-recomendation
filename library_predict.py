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
import relation_model
from load import load_dataset
import pickle
import csv


# Calculate the hit rate of top predicted libraries
def calculate_hit_rate(actual_libraries, top_libraries_predicted):
    predicted_count = 0
    for library_predicted in top_libraries_predicted:
        if library_predicted in actual_libraries:
            predicted_count = predicted_count + 1
    hit_rate = predicted_count / len(top_libraries_predicted)
    return hit_rate


# Calculate the similarity for each library with every keyword
def calculate_similarity(libraries, embeddings, lib_key_graph, path_keywords, idf_dict, method="dot",
                         idf="False"):
    nodes = lib_key_graph.nodes()
    # Initialize a dictionary with 0 for the cos similarity of every library
    sim = {library: 0 for library in libraries}
    #if idf == "True":
    #    idf_dict = tokenize_files.tf_idf(train_keywords, file_paths)
    path_keywords = [keyword for keyword in path_keywords if idf_dict.get(keyword, 0)<1]
    posistional_importance = {keyword: 1.5**position for position, keyword in enumerate(path_keywords)}
    if method == "cosine":
        for library in libraries:
            for keyword in path_keywords: #nx.ego_graph(lib_key_graph, 'lib:'+library,1):
                if keyword in nodes and 'lib:' + library in nodes:
                    lib_array = embeddings['lib:' +library]
                    keyword_array = embeddings[keyword]
                    #if keyword not in nx.ego_graph(lib_key_graph, 'lib:'+library,3): #lib_key_graph._adj['lib:'+library]:
                    #    continue
                    cos_similarity =np.dot(lib_array, keyword_array) / (norm(lib_array) * norm(keyword_array)) *posistional_importance[keyword]
                    if idf == "True":
                        if keyword in idf_dict.keys():
                            sim[library] = sim[library] + cos_similarity * idf_dict.get(keyword,1)/idf_dict.get(library,1)
                        else:
                            sim[library] = sim[library] + cos_similarity
                    else:
                        sim[library] = sim[library] + cos_similarity
    elif method == "dot":
        for library in libraries:
            for keyword in path_keywords:
                if keyword in nodes and 'lib:'+ library in nodes:
                    lib_array = embeddings['lib:' +library]
                    keyword_array = embeddings[keyword]
                    dot_similarity = np.dot(lib_array, keyword_array)*posistional_importance[keyword]
                    if idf == "True":
                        if keyword in idf_dict.keys():
                            sim[library] = sim[library] + dot_similarity * idf_dict.get(keyword,1)/idf_dict.get(library,1)
                        else:
                            sim[library] = sim[library] + dot_similarity
                    else:
                        sim[library] = sim[library] + dot_similarity

    return sim

def caclulate_function_similarity(libraries, embeddings, lib_key_graph, path_keywords, scaler, model, idf_dict,
                         idf="False"):
    nodes = lib_key_graph.nodes()
    # Initialize a dictionary with 0 for the cos similarity of every library
    sim = {library: 0 for library in libraries}
    path_keywords = [keyword for keyword in path_keywords if idf_dict.get(keyword, 0) < 1]
    posistional_importance = {keyword: 1.5 ** position for position, keyword in enumerate(path_keywords)}
    for library in libraries:
        for keyword in path_keywords:
            if keyword in nodes and 'lib:' + library in nodes:
                lib_features = scaler.transform([embeddings['lib:' +library]])
                key_features = scaler.transform([embeddings[keyword]])
                function_similarity = model.predict(np.multiply(lib_features, key_features))
                if idf == "True":
                    if keyword in idf_dict.keys():
                        sim[library] = sim[library] + int(function_similarity[0]) * idf_dict.get(keyword,1)/idf_dict.get(library,1) *posistional_importance[keyword]
                    else:
                        sim[library] = sim[library] + int(function_similarity[0])
                else:
                    sim[library] = sim[library] + int(function_similarity[0])
    return sim


# Predict the values for the libraries in this file using the adjacency matrix and an graph signal
# with 1s where keyword exists and 0 where keyword doesn't
def  predict_values_with_graph(path_keywords, training_graph):
    adj_matrix = nx.to_numpy_matrix(training_graph)
    node_atr = list(repeat(0, len(adj_matrix)))
    nodes_names = list(training_graph.nodes())
    for keyword in path_keywords:
        if keyword in nodes_names:
            node_atr[nodes_names.index(keyword)] = 1
    graph_filter = np.matmul(adj_matrix, np.asarray(node_atr))
    value_list = graph_filter.tolist()[0]
    return value_list


# Return the embeddings of the graph using the L.I.N.E. or teh node
def get_embeddings(training_graph, embedddings_method="line", proximity_method="both"):
    if embedddings_method == "line":
        if proximity_method == "both":
            args = Namespace(embedding_dim=16, batch_size=128, K=5, proximity="first-order", learning_rate=0.025,
                             mode="train", num_batches=1000, total_graph=True,
                             graph_file="line_algo/data/keras-graph.gpickle")
            # Dictionary with the embedding of every library using L.I.N.E. algorithm
            embeddings_first = line.train(args)

            args = Namespace(embedding_dim=16, batch_size=128, K=5, proximity="second-order", learning_rate=0.025,
                             mode="train", num_batches=1000, total_graph=True,
                             graph_file="line_algo/data/keras-graph.gpickle")
            embeddings_second = line.train(args)
            embeddings = {}
            for node in training_graph.nodes():
                embeddings[node] = np.concatenate((embeddings_first[node], 0.3 * embeddings_second[node]), axis=None)
        elif proximity_method == "first":
            args = Namespace(embedding_dim=16, batch_size=16, K=5, proximity="first-order", learning_rate=0.025,
                             mode="train", num_batches=1000, total_graph=True,
                             graph_file="line_algo/data/keras-graph.gpickle")
            # Dictionary with the embedding of every library using L.I.N.E. algorithm
            embeddings = line.train(args)
        elif proximity_method == "second":
            args = Namespace(embedding_dim=16, batch_size=16, K=5, proximity="second-order", learning_rate=0.025,
                             mode="train", num_batches=1000, total_graph=True,
                             graph_file="line_algo/data/keras-graph.gpickle")
            # Dictionary with the embedding of every library using L.I.N.E. algorithm
            embeddings = line.train(args)
    elif embedddings_method == "node2vec":
        # Precompute probabilities and generate walks - **ON WINDOWS ONLY WORKS WITH workers=1**
        node2vec = Node2Vec(training_graph, dimensions=16, walk_length=80, num_walks=20,
                            workers=1, temp_folder="./temp_graph", p=2, q=0.5)  # Use temp_folder for big graphs
        # Embed nodes
        model = node2vec.fit(window=4, min_count=1, batch_words=4)

        embeddings = {node: model.wv[node] for node in training_graph.nodes}
        model.save('./embeddings.model')

    return embeddings


def predict_without_embeddings(training_graph, libraries, path_keywords, random_predict="False", k=5):
    if random_predict=="True":
        sim = {library: random.randint(1, 1000) for library in libraries}
        predicted_libraries = nlargest(k, sim, key=sim.get)
    else:
        # Predict libraries in this path
        lib_predicted_values = predict_values_with_graph(path_keywords, training_graph)
        nodes_names = list(training_graph.nodes())
        sim = {}
        for lib_value in lib_predicted_values:
            if nodes_names[lib_predicted_values.index(lib_value)] in libraries:
                sim[nodes_names[lib_predicted_values.index(lib_value)]] = lib_value

        predicted_libraries = nlargest(k, sim, key=sim.get)

    return predicted_libraries, sim


def test_files_with_embeddings(test_set, file_paths, libraries, keywords, similarity_method, training_graph,
                               embeddings):
    if similarity_method == 'function':
        # Create training set for the model of the similarity prediction
        training_features, training_values = relation_model.create_training_set(training_graph, embeddings,
                                                                                libraries, keywords)
        # Train the relation model
        scaler, model = relation_model.train_relation_model(training_features, training_values)

    idf_dict = tokenize_files.tf_idf_files(file_paths)
    hit_rate = []
    auc = []
    ndcg = []
    # Store the results in a file
    f = open("results.txt", "w")
    for file_path in test_set:
        # Get the path's libraries and keyword for the specific file in the path
        print('Predict path: ', file_path)
        path_libraries, path_keywords = tokenize_files.get_libs_and_keywords(file_path)
        print("Number of libraries in this file: ", len(path_libraries))

        # Calculate similarity and save it in a dictionary
        if similarity_method == "function":
            sim = caclulate_function_similarity(libraries, embeddings, training_graph, path_keywords, scaler, model, idf_dict,
                                           idf="True")
        else:
            sim = calculate_similarity(libraries, embeddings, training_graph, path_keywords, idf_dict, method=similarity_method,
                                 idf="True")

        # Get the largest 5 values
        predicted_libraries = nlargest(10, sim, key=sim.get)
        print("Libraries predicted: ", predicted_libraries)
        print("Path libraries:", path_libraries, "\n")

        # Hit rate for Top-5 libraries
        hit_rate_temp = calculate_hit_rate(path_libraries, predicted_libraries)
        hit_rate.append(hit_rate_temp)
        print("Hit Rate @", len(predicted_libraries), ": ", hit_rate_temp)
        f.write("Hit Rate: ")
        f.write(str(hit_rate_temp) + "\n")
        # Calculate AUC
        labels = [1 if library in path_libraries else 0 for library in sim.keys()]
        conf = list(sim.values())
        auc_temp = roc_auc_score(np.array(labels), np.array(conf))
        auc.append(auc_temp)
        print("ROC AUC: ", auc_temp, "\n")
        f.write("AUC: ")
        f.write(str(auc_temp) + "\n")
        # Calculate Normalized Cumulative Score
        # Relevance score=1 if a library that was predicted is in path's libraries
        ndcg.append(ndcg_score([np.array(labels)], [np.array(conf)]))
        print("Discounted Cumulative Gain: ", ndcg_score([np.array(labels)], [np.array(conf)]), '\n')
    f.close()

    return predicted_libraries, auc, ndcg, hit_rate

def load_model_data():

    with open('idf.data', 'rb') as filehandle:
        # store the data as binary data stream
        idf_dict = pickle.load(filehandle)

    with open('libraries.data', 'rb') as filehandle:
        # store the data as binary data stream
        libraries = pickle.load(filehandle)

    with open('keywords.data', 'rb') as filehandle:
        # store the data as binary data stream
        keywords = pickle.load(filehandle)

    with open('training_libs.txt', 'rb') as filehandle:
        # store the data as binary data stream
        train_domains_libraries = pickle.load(filehandle)

    with open('training_keys.txt', 'rb') as filehandle:
        # store the data as binary data stream
        train_domains_keywords = pickle.load(filehandle)

    with open('testing_libs.txt', 'rb') as filehandle:
        # store the data as binary data stream
        test_domains_libraries = pickle.load(filehandle)

    with open('testing_keywords.txt', 'rb') as filehandle:
        # store the data as binary data stream
        test_domains_keywords = pickle.load(filehandle)

    return libraries, keywords, test_domains_libraries, test_domains_keywords, \
           train_domains_libraries, train_domains_keywords, idf_dict


def predict_libraries():

    dataset_name = "keras"
    if dataset_name == "keras":
        libraries, keywords, lib_key_graph, test_domains_libraries, test_domains_keywords, \
        train_domains_libraries, train_domains_keywords = load_keras()
    else:
        libraries, keywords, lib_key_graph, test_domains_libraries, test_domains_keywords, train_domains_libraries, \
        train_domains_keywords = load_dataset(number_of_methods=50000, num_of_keywords_after_dot=0)


    # Store graph in a file
    nx.write_gpickle(lib_key_graph, "line_algo/data/keras-graph.gpickle")

    '''
    libraries2, keywords2, test_domains_libraries2, test_domains_keywords2, train_domains_libraries2, \
    train_domains_keywords2 = load_model_data()
    '''
    embeddings_method = "line"
    have_embeddings = "True"
    random_predict = "False"
    similarity_methods = ['cosine', 'dot', 'function']
    proximities = ['first', 'second']
    idf_uses = ['True']

    '''
    if have_embeddings == "False":
        for domain in test_domains_libraries.keys():

            hit_rate = []
            auc = []
            ndcg = []

            libraries_predicted_list=[]
            # Check if libraries was identified in this domain
            if len(test_domains_libraries[domain]) > 0:
                # Get the path's libraries and keyword for the specific file in the path
                print('Predict path: ', domain)
                print("Number of libraries in this file: ", len(test_domains_libraries[domain]))

                path_keywords = test_domains_keywords[domain]
                path_libraries = test_domains_libraries[domain]

                # Get the largest k recommended libraries
                predicted_libraries, sim = predict_without_embeddings(lib_key_graph, libraries, path_keywords,
                                                                      random_predict, k=10)

                libraries_predicted_list = libraries_predicted_list + predicted_libraries

                print("Libraries predicted: ", predicted_libraries)
                print("Path libraries:", path_libraries, "\n")
                path_libraries.sort()

                # Hit rate for Top-5 libraries
                hit_rate_temp = calculate_hit_rate(path_libraries, predicted_libraries)
                hit_rate.append(hit_rate_temp)
                print("Hit Rate @", len(predicted_libraries), ": ", hit_rate_temp)

                # Calculate AUC
                labels = [1 if library in path_libraries else 0 for library in sim.keys()]
                conf = list(sim.values())
                if 1 in labels:
                    auc.append(roc_auc_score(np.array(labels), np.array(conf)))
                    print("ROC AUC: ", roc_auc_score(np.array(labels), np.array(conf)), "\n")

                # Calculate Normalized Cumulative Score
                # Relevance score=1 if a library that was predicted is in path's libraries
                ndcg.append(ndcg_score([np.array(labels)], [np.array(conf)]))
                print("Discounted Cumulative Gain: ", ndcg_score([np.array(labels)], [np.array(conf)]), '\n')
    else:
    '''

    results = open('results.csv', mode='w')
    results = csv.writer(results, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    results.writerow(["1st Prox.", "2st Prox.", 'idf', "Similarity", "HitRate@10", "AUC", "NDCG", "Coverage"])

    idf_dict = {**tokenize_files.tf_idf(train_domains_keywords), **tokenize_files.tf_idf(train_domains_libraries)}


    for proximity in proximities:
        for idf_use in idf_uses:
            for similarity_method in similarity_methods:
                results.writerow([proximity, similarity_method, idf_use])

                coverage = []
                hit_rate = []
                auc = []
                ndcg = []
                # Store the results in a file
                for i in range(10):

                    if similarity_method == 'function':
                        # Create training set for the model of the similarity prediction
                        training_features, training_values = relation_model.create_training_set(lib_key_graph,
                                                                                                embeddings,
                                                                                                libraries, keywords)
                        # Train the relation model
                        scaler, model = relation_model.train_relation_model(training_features, training_values)

                    embeddings = get_embeddings(lib_key_graph, embeddings_method, proximity_method=proximity)

                    libraries_predicted_list = []

                    for domain in test_domains_libraries.keys():
                        # Check if libraries was identified in this domain
                        if len(test_domains_libraries[domain]) >= 0 : #and len(test_domains_libraries[domain])> 5:

                            path_keywords = test_domains_keywords[domain]
                            path_libraries = test_domains_libraries[domain]
                            print('Predict path: ', domain)
                            print("Number of libraries in this file: ", len(test_domains_libraries[domain]))

                            # Calculate similarity and save it in a dictionary
                            if similarity_method == "function":
                                sim = caclulate_function_similarity(libraries, embeddings, lib_key_graph, path_keywords, scaler,
                                                                    model,
                                                                    idf_dict, idf=idf_use)
                            else:
                                sim = calculate_similarity(libraries, embeddings, lib_key_graph, path_keywords, idf_dict,
                                                           method=similarity_method, idf=idf_use)
                            # print(sim)
                            # Get the largest 5 values
                            predicted_libraries = nlargest(10, sim, key=sim.get)
                            print("Libraries predicted: ", predicted_libraries)
                            print("Path libraries:", path_libraries, "\n")

                            libraries_predicted_list = libraries_predicted_list + predicted_libraries
                            for library in predicted_libraries:
                                if library in path_libraries:
                                    print(library)
                            # Hit rate for Top-5 libraries
                            hit_rate_temp = calculate_hit_rate(path_libraries, predicted_libraries)
                            hit_rate.append(hit_rate_temp)
                            print("Hit Rate @", len(predicted_libraries), ": ", hit_rate_temp)
                            # Calculate AUC
                            labels = [1 if library in path_libraries else 0 for library in sim.keys()]
                            conf = list(sim.values())
                            if 1 in labels and 0 in labels:
                                auc_temp = roc_auc_score(np.array(labels), np.array(conf))
                                auc.append(auc_temp)
                                print("ROC AUC: ", auc_temp, "\n")
                            # Calculate Normalized Cumulative Score
                            # Relevance score=1 if a library that was predicted is in path's libraries
                            ndcg_temp = ndcg_score([np.array(labels)], [np.array(conf)])
                            ndcg.append(ndcg_temp)
                            print("Discounted Cumulative Gain: ", ndcg_score([np.array(labels)], [np.array(conf)]), '\n')
                    libraries_predicted_list = list(set(libraries_predicted_list))
                    coverage.append(len(libraries_predicted_list) / len(libraries))
                results.writerow([sum(hit_rate) / len(hit_rate), sum(auc) / len(auc), sum(ndcg) / len(ndcg),
                                                            len(libraries_predicted_list) / len(libraries) * 100])
            #results.writerow([sum(hit_rate) / len(hit_rate), sum(auc) / len(auc), sum(ndcg) / len(ndcg),
            #                  sum(coverage)/len(coverage)*100])

    libraries.sort()
    print(libraries)

    print(" \n Hit rate @", len(predicted_libraries), " \n        Average: ", sum(hit_rate) / len(hit_rate),
          " \n        STD: ", np.std(hit_rate))
    print("        Range: ", min(hit_rate), " - ", max(hit_rate))

    print(" \n AUC", " \n        Average: ", sum(auc) / len(auc),
          " \n        STD: ", np.std(auc))
    print("        Range: ", min(auc), " - ", max(auc))

    print(" \n NDCG", " \n        Average: ", sum(ndcg) / len(ndcg),
          " \n        STD: ", np.std(ndcg))
    print("        Range: ", min(ndcg), " - ", max(ndcg))


def load_keras(graph_method = "co-occur"):

    # Get all the paths inside the directory path provided before
    file_paths = graph_creator.get_all_paths('keras\\tests')

    # Training 80% Testing 20%
    training_set, test_set = train_test_split(file_paths, test_size=0.2, random_state=0)

    test_domains_keywords = {}
    test_domains_libraries = {}

    # Create the graph of the training set with the appropriate method
    if graph_method == "co-occur":
        libraries, keywords, lib_key_graph, train_domains_libraries, train_domains_keywords = graph_creator.create_graph(training_set)
    elif graph_method == "line-by-line":
        libraries, keywords, training_graph = line_by_line_graph(training_set)

    # Store libraries and keywords of each fie of the test set
    for file_path in test_set:
        path_libraries, path_keywords = tokenize_files.get_libs_and_keywords(file_path, double_keywords_held=False)
        test_domains_libraries[file_path] = path_libraries
        test_domains_keywords[file_path] = path_keywords

    return libraries, keywords, lib_key_graph, test_domains_libraries, test_domains_keywords, \
           train_domains_libraries, train_domains_keywords


if __name__ == "__main__":
    predict_libraries()
    '''
    dataset_name = "projects"
    if dataset_name == "keras":
        libraries, keywords, lib_key_graph, test_domains_libraries, test_domains_keywords, \
        train_domains_libraries, train_domains_keywords = load_keras()
    else:
        libraries, keywords, lib_key_graph, test_domains_libraries, test_domains_keywords, train_domains_libraries, \
        train_domains_keywords = load_dataset(number_of_methods=100000, num_of_keywords_after_dot=0)

    # Store graph in a file
    nx.write_gpickle(lib_key_graph, "line_algo/data/lib_rec.gpickle")

    embeddings_method = "line"
    have_embeddings = "True"
    random_predict = "False"
    similarity_method = 'cosine'
    proximity = 'both'
    idf_use = 'False'

    embeddings = get_embeddings(lib_key_graph, embeddings_method, proximity_method=proximity)
    idf_dict = tokenize_files.tf_idf(train_domains_keywords)
    path_keywords = test_domains_keywords[5]
    for key in path_keywords:
        sim = calculate_similarity(libraries, embeddings, lib_key_graph, [key], idf_dict,
                                   method=similarity_method, idf=idf_use)
        predicted_libraries = nlargest(10, sim, key=sim.get)
        print("Keyword: ", key)
        print(predicted_libraries)

        #path_libraries = test_domains_libraries[domain]
    '''