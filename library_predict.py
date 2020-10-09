import random
import graph_creator
import networkx as nx
import tokenize_files
import numpy as np
from numpy.linalg import norm
from itertools import repeat
from heapq import nlargest
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import ndcg_score
from line_algo import line
from argparse import Namespace
from sklearn.preprocessing import StandardScaler
from node2vec import Node2Vec
from gensim.models import Word2Vec
from graph_line_by_line import line_by_line_graph
import relation_model
import pickle
import csv
from load import load_dataset


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
    # if idf == "True":
    #    idf_dict = tokenize_files.tf_idf(train_keywords, file_paths)
    path_keywords = [keyword for keyword in path_keywords if idf_dict.get(keyword, 0) < 1]
    posistional_importance = {keyword: 1.5 ** position for position, keyword in enumerate(path_keywords)}
    if method == "cosine":
        for library in libraries:
            for keyword in path_keywords:  # nx.ego_graph(lib_key_graph, 'lib:'+library,1):
                if keyword in nodes and 'lib:' + library in nodes:
                    lib_array = embeddings['lib:' + library]
                    keyword_array = embeddings[keyword]
                    # if keyword not in nx.ego_graph(lib_key_graph, 'lib:'+library,3): #lib_key_graph._adj['lib:'+library]:
                    #    continue
                    cos_similarity = np.dot(lib_array, keyword_array) / (norm(lib_array) * norm(keyword_array)) * \
                                     posistional_importance[keyword]
                    if idf == "True":
                        if keyword in idf_dict.keys():
                            sim[library] = sim[library] + cos_similarity * idf_dict.get(keyword, 1) / idf_dict.get(
                                library, 1)
                        else:
                            sim[library] = sim[library] + cos_similarity
                    else:
                        sim[library] = sim[library] + cos_similarity
    elif method == "dot":
        for library in libraries:
            for keyword in path_keywords:
                if keyword in nodes and 'lib:' + library in nodes:
                    lib_array = embeddings['lib:' + library]
                    keyword_array = embeddings[keyword]
                    dot_similarity = np.dot(lib_array, keyword_array) * posistional_importance[keyword]
                    if idf == "True":
                        if keyword in idf_dict.keys():
                            sim[library] = sim[library] + dot_similarity * idf_dict.get(keyword, 1) / idf_dict.get(
                                library, 1)
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
                lib_features = scaler.transform([embeddings['lib:' + library]])
                key_features = scaler.transform([embeddings[keyword]])
                function_similarity = model.predict(np.multiply(lib_features, key_features))
                if idf == "True":
                    if keyword in idf_dict.keys():
                        sim[library] = sim[library] + int(function_similarity[0]) * idf_dict.get(keyword,
                                                                                                 1) / idf_dict.get(
                            library, 1) * posistional_importance[keyword]
                    else:
                        sim[library] = sim[library] + int(function_similarity[0])
                else:
                    sim[library] = sim[library] + int(function_similarity[0])
    return sim


# Predict the values for the libraries in this file using the adjacency matrix and an graph signal
# with 1s where keyword exists and 0 where keyword doesn't
def predict_values_with_graph(path_keywords, training_graph):
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
def get_embeddings(training_graph, graph_name='lib_rec', embeddings_method="line", proximity_method="both", ):
    if embeddings_method == "line":
        if proximity_method == "both":
            args = Namespace(embedding_dim=16, batch_size=128, K=5, proximity="first-order", learning_rate=0.025,
                             mode="train", num_batches=1000, total_graph=True,
                             graph_file="line_algo/data/" + graph_name + ".gpickle")
            # Dictionary with the embedding of every library using L.I.N.E. algorithm
            embeddings_first = line.train(args)

            args = Namespace(embedding_dim=16, batch_size=128, K=5, proximity="second-order", learning_rate=0.025,
                             mode="train", num_batches=1000, total_graph=True,
                             graph_file="line_algo/data/" + graph_name + ".gpickle")
            embeddings_second = line.train(args)
            embeddings = {}
            for node in training_graph.nodes():
                embeddings[node] = np.concatenate((embeddings_first[node], 0.3 * embeddings_second[node]), axis=None)
        elif proximity_method == "first":
            args = Namespace(embedding_dim=16, batch_size=16, K=5, proximity="first-order", learning_rate=0.025,
                             mode="train", num_batches=1000, total_graph=True,
                             graph_file="line_algo/data/" + graph_name + ".gpickle")
            # Dictionary with the embedding of every library using L.I.N.E. algorithm
            embeddings = line.train(args)
        elif proximity_method == "second":
            args = Namespace(embedding_dim=16, batch_size=16, K=5, proximity="second-order", learning_rate=0.025,
                             mode="train", num_batches=1000, total_graph=True,
                             graph_file="line_algo/data/" + graph_name + ".gpickle")
            # Dictionary with the embedding of every library using L.I.N.E. algorithm
            embeddings = line.train(args)
    elif embeddings_method == "node2vec":
        # Precompute probabilities and generate walks - **ON WINDOWS ONLY WORKS WITH workers=1**
        node2vec = Node2Vec(training_graph, dimensions=16, walk_length=80, num_walks=20,
                            workers=1, temp_folder="./temp_graph", p=2, q=0.5)  # Use temp_folder for big graphs
        # Embed nodes
        model = node2vec.fit(window=4, min_count=1, batch_words=4)

        embeddings = {node: model.wv[node] for node in training_graph.nodes}
        model.save('line_algo/data/embeddings_node2vec.model')

    return embeddings


def predict_without_embeddings(training_graph, libraries, test_domains_libraries, test_domains_keywords,
                               random_predict="False"):
    for domain in test_domains_libraries.keys():

        hit_rate = []
        auc = []
        ndcg = []
        libraries_predicted_list = []
        # Check if libraries was identified in this domain
        if len(test_domains_libraries[domain]) > 0:
            # Get the path's libraries and keyword for the specific file in the path
            print('Predict path: ', domain)
            print("Number of libraries in this file: ", len(test_domains_libraries[domain]))

            path_keywords = test_domains_keywords[domain]
            path_libraries = test_domains_libraries[domain]

            # Get the largest k recommended libraries
            if random_predict == "True":
                sim = {library: random.randint(1, 1000) for library in libraries}
                predicted_libraries = nlargest(10, sim, key=sim.get)
            else:
                # Predict libraries in this path
                lib_predicted_values = predict_values_with_graph(path_keywords, training_graph)
                nodes_names = list(training_graph.nodes())
                sim = {}
                for lib_value in lib_predicted_values:
                    if nodes_names[lib_predicted_values.index(lib_value)] in libraries:
                        sim[nodes_names[lib_predicted_values.index(lib_value)]] = lib_value

                predicted_libraries = nlargest(10, sim, key=sim.get)

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

    return predicted_libraries, sim


def load_model_data():
    with open('data/idf.data', 'rb') as filehandle:
        # store the data as binary data stream
        idf_dict = pickle.load(filehandle)

    with open('data/libraries.data', 'rb') as filehandle:
        # store the data as binary data stream
        libraries = pickle.load(filehandle)

    with open('data/keywords.data', 'rb') as filehandle:
        # store the data as binary data stream
        keywords = pickle.load(filehandle)

    return libraries, keywords, idf_dict


def train_model():
    libraries, keywords, lib_key_graph, test_domains_libraries, test_domains_keywords, train_domains_libraries, \
    train_domains_keywords = load_dataset(number_of_methods=100000, num_of_keywords_after_dot=0)


if __name__ == "__main__":
    train_model()