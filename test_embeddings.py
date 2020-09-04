import graph_creator
import networkx as nx
import matplotlib.pyplot as plt
import tokenize_files
import numpy as np
from numpy.linalg import norm
from heapq import nlargest
from line_algo import line
from argparse import Namespace
from library_predict import calculate_similarity
from library_predict import load_model_data
from library_predict import calculate_hit_rate
from library_predict import caclulate_function_similarity
from load import get_libs_and_keywords_file
from heapq import nlargest
from load import load
from load import load_train_method
import relation_model
import csv
from sklearn.metrics import roc_auc_score
from sklearn.metrics import ndcg_score


def test_experiment(libraries, keywords, idf_dict, lib_key_graph, number_of_methods=100000):
    methods = load(limit=number_of_methods)

    method_list = []
    for method in methods:
        method_list.append(method)
    predict_by_project = 'True'
    dataset_length = len(method_list)
    counter = 1
    train_domains = []
    test_domains_keywords = {}
    test_domains_libraries = {}
    for method in method_list:
        if counter <= 0.8 * dataset_length:
            train_domains.append(method.repo)
        else:
            if method.repo not in train_domains:

                # Load libraries and keywords for the method
                actual_libraries, actual_keywords = load_train_method(method, libraries, num_of_keywords_after_dot=0)

                if predict_by_project == "True":
                    # Store libraries and keywords for each project into a dictionary
                    if method.repo not in test_domains_libraries.keys():
                        test_domains_libraries[method.repo] = actual_libraries
                        test_domains_keywords[method.repo] = actual_keywords
                    else:
                        test_domains_libraries[method.repo] = list(
                            set(test_domains_libraries[method.repo] + actual_libraries))
                        test_domains_keywords[method.repo] = list(
                            set(test_domains_keywords[method.repo] + actual_keywords))
                else:
                    test_domains_libraries[method.name] = actual_libraries
                    test_domains_keywords[method.name] = actual_keywords

        counter = counter + 1

    embeddings_method = "line"
    have_embeddings = "True"
    random_predict = "False"
    similarity_methods = ['cosine', 'dot', 'function']
    proximities = ['first', 'second']
    idf_uses = ['True']

    embeddings_first = nx.read_gpickle('line_algo\data\embedding_first-order.gpickle')
    embeddings_second = nx.read_gpickle('line_algo\data\embedding_second-order.gpickle')

    results = open('results.csv', mode='w')
    results = csv.writer(results, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    results.writerow(["1st Prox.", "2st Prox.", 'idf', "Similarity", "HitRate@10", "AUC", "NDCG", "Coverage"])
    for proximity in proximities:
        for idf_use in idf_uses:
            for similarity_method in similarity_methods:

                embeddings = {}
                for node in G.nodes():
                    if proximity == 'first':
                        embeddings[node] = embeddings_first[node]
                    elif proximity == 'second':
                        embeddings[node] = embeddings_second[node]
                    else:
                        embeddings[node] = np.concatenate((embeddings_first[node], 0.3 * embeddings_second[node]),
                                                          axis=None)

                results.writerow([proximity, similarity_method, idf_use])
                coverage = []
                hit_rate = []
                auc = []
                ndcg = []
                #embeddings = get_embeddings(lib_key_graph, embeddings_method, proximity_method=proximity)

                if similarity_method == 'function':
                    # Create training set for the model of the similarity prediction
                    training_features, training_values = relation_model.create_training_set(lib_key_graph, embeddings,
                                                                                            libraries, keywords)
                    # Train the relation model
                    scaler, model = relation_model.train_relation_model(training_features, training_values)

                libraries_predicted_list = []
                # Store the results in a file
                for domain in test_domains_libraries.keys():
                    # Check if libraries was identified in this domain
                    if len(test_domains_libraries[domain]) >= 0 and len(test_domains_libraries[domain]) > 5:

                        path_keywords = test_domains_keywords[domain]
                        path_libraries = test_domains_libraries[domain]
                        print('Predict path: ', domain)
                        print("Number of libraries in this file: ", len(test_domains_libraries[domain]))

                        # Calculate similarity and save it in a dictionary
                        if similarity_method == "function":
                            sim = caclulate_function_similarity(libraries, embeddings, lib_key_graph, path_keywords,
                                                                scaler,
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
                results.writerow([sum(hit_rate) / len(hit_rate), sum(auc) / len(auc), sum(ndcg) / len(ndcg), len(libraries_predicted_list)/len(libraries)*100])
                coverage.append(len(libraries_predicted_list) / len(libraries))
            # results.writerow([sum(hit_rate) / len(hit_rate)*100, sum(auc) / len(auc)*100, sum(ndcg) / len(ndcg)*100,
            #                  sum(coverage)/len(coverage)*100])





def get_queries():
    test_domains_keywords = {}
    test_domains_libraries = {}
    _, test_domains_keywords[1] = get_libs_and_keywords_file('Calling an external command', double_keywords_held=True,
                                                             dot_break=False, stem_use="True")  # subprocess , os, shlex
    test_domains_libraries[1] = ['subprocess', 'os', 'shlex']
    _, test_domains_keywords[2] = get_libs_and_keywords_file('How to sort a dictionary by value',
                                                             double_keywords_held=True, dot_break=False,
                                                             stem_use="True")  # operator, collections , operator.itemgetter
    test_domains_libraries[2] = ['operator', 'collections']
    _, test_domains_keywords[3] = get_libs_and_keywords_file('How to convert string representation of list to a list',
                                                             double_keywords_held=False, dot_break=False,
                                                             stem_use="True")  # ast, json, re, pyparsing
    test_domains_libraries[3] = ['ast', 'json', 're', 'pyparsing']
    _, test_domains_keywords[4] = get_libs_and_keywords_file('How do i merge two dictionaries in a single expression',
                                                             double_keywords_held=True, dot_break=False,
                                                             stem_use="True")  # collections.ChainMap, itertools.chain, copy
    test_domains_libraries[4] = ['collections', 'itertools', 'copy']
    _, test_domains_keywords[5] = get_libs_and_keywords_file('Converting string into datetime',
                                                             double_keywords_held=True, dot_break=False,
                                                             stem_use="True")  # datetime ,dateutil , timestring
    test_domains_libraries[5] = ['datetime', 'dateutil', 'timestring']
    _, test_domains_keywords[6] = get_libs_and_keywords_file('How to import a module given the full path',
                                                             double_keywords_held=True, dot_break=False,
                                                             stem_use="True",
                                                             return_libraries=False)  # imp, importlib.util, runpy , pkgutil
    test_domains_libraries[6] = ['imp', 'importlib', 'runpy', 'pkgutil']
    _, test_domains_keywords[7] = get_libs_and_keywords_file('How to get current time', double_keywords_held=True,
                                                             dot_break=False,
                                                             stem_use="True")  # time, datetime , pandas, numpy, arrow
    test_domains_libraries[7] = ['time', 'datetime', 'pandas', 'numpy', 'arrow']
    _, test_domains_keywords[8] = get_libs_and_keywords_file('How do I concatenate two lists',
                                                             double_keywords_held=True, dot_break=False,
                                                             stem_use="True")  # itertools, heapq, operator
    test_domains_libraries[8] = ['itertools', 'heapq', 'operator']
    _, test_domains_keywords[9] = get_libs_and_keywords_file('How to count occurrences of a list item',
                                                             double_keywords_held=True, dot_break=False,
                                                             stem_use="True")  # numpy, more_itertools, collections
    test_domains_libraries[9] = ['numpy', 'more_itertools', 'collections']
    _, test_domains_keywords[10] = get_libs_and_keywords_file('How do I download a file over HTTP',
                                                              double_keywords_held=True, dot_break=False,
                                                              stem_use="True")  # urllib, requests, wget, pycurl
    test_domains_libraries[10] = ['urllib', 'requests', 'wget', 'pycurl']
    _, test_domains_keywords[11] = get_libs_and_keywords_file('How to use threading', double_keywords_held=True,
                                                              dot_break=False,
                                                              stem_use="True")  # threading, multiprocessing, concurrent
    test_domains_libraries[11] = ['threading', 'multiprocessing', 'concurrent']
    _, test_domains_keywords[12] = get_libs_and_keywords_file('how to connect to mysql database',
                                                              double_keywords_held=True, dot_break=False,
                                                              stem_use="True")  # MySQLdb, peewee, mysql.connector, pymysql, oursql, flask_mysqldb
    test_domains_libraries[12] = ['MySQLdb', 'peewee', 'mysql', 'pymysql', 'oursql', 'flask_mysqldb']
    _, test_domains_keywords[13] = get_libs_and_keywords_file('how to write json data to a file',
                                                              double_keywords_held=True, dot_break=False,
                                                              stem_use="True")  # json, mpu.io
    test_domains_libraries[13] = ['json', 'mpu']
    _, test_domains_keywords[14] = get_libs_and_keywords_file('How to get POSTed JSON in Flask',
                                                              double_keywords_held=True, dot_break=False,
                                                              stem_use="True")  # flask, requests, json
    test_domains_libraries[14] = ['flask', 'requests', 'json']

    return test_domains_keywords, test_domains_libraries


G = nx.read_gpickle("line_algo/data/lib_rec.gpickle")

# print(list(G.neighbors('datetim')))

libraries, keywords, test_domains_libraries, test_domains_keywords, \
train_domains_libraries, train_domains_keywords, idf_dict = load_model_data()
'''
#test_experiment(libraries, keywords, idf_dict, G)








embeddings_first = nx.read_gpickle('line_algo\data\embedding_first-order.gpickle')
embeddings_second = nx.read_gpickle('line_algo\data\embedding_second-order.gpickle')

embeddings = {}
for node in G.nodes():
    embeddings[node] = np.concatenate((embeddings_first[node], 0.3 * embeddings_second[node]), axis=None)

# print(np.dot(embeddings['lib:time'], embeddings['lib:datetime'] / (norm(embeddings['lib:time']) * norm(embeddings['lib:datetime']))))
# test_string = 'How to use threading'
# _, path_keywords = get_libs_and_keywords_file(test_string, double_keywords_held=True, dot_break=False, stem_use="True",
#                                              return_libraries=False)
libraries = [node[4:] for node in G if 'lib:' in node]
queries_keywords, queries_libraries = get_queries()

useful_libraries = 0
for domain in queries_keywords.keys():
    path_keywords = queries_keywords[domain]
    path_libraries = queries_libraries[domain]
    sim = calculate_similarity(libraries, embeddings, G, path_keywords, idf_dict, method='cosine', idf='True')
    predicted_libraries = nlargest(10, sim, key=sim.get)
    for lib in predicted_libraries:
        if lib in path_libraries:
            useful_libraries = useful_libraries + 1
    print("Path keywords: ", path_keywords)
    print("Path libraries: ", path_libraries)
    print("Libraries predicted: ", predicted_libraries, "\n")

print("Number of useful libraries: ", useful_libraries)

libs = []
for node in G:
    if 'lib:' in node:
        libs.append(node[4:])
libs.sort()
print(libs)
print("Total number of libraries", len(libs))
print("Total number of keywords", len(keywords))


_, path_keywords = get_libs_and_keywords_file("How to convert string representation of list to a list", double_keywords_held=False,
                                                             dot_break=False, stem_use="True")
print(path_keywords)
for key in path_keywords:
    sim = calculate_similarity(libraries, embeddings, G, [key], idf_dict, method='cosine', idf='True')
    predicted_libraries = nlargest(10, sim, key=sim.get)
    print("Keyword: ", key)
    print(idf_dict[key])
    print(predicted_libraries)
'''
methods = load(limit=100000)

method_list = []
for method in methods:
    method_list.append(method)
predict_by_project = 'True'
dataset_length = len(method_list)
counter = 1
train_domains = []
test_domains_keywords = {}
test_domains_libraries = {}
for method in method_list:
    if counter <= 0.8 * dataset_length:
        if method.repo not in train_domains:
            train_domains.append(method.repo)
    else:
        if method.repo not in train_domains:

            # Load libraries and keywords for the method
            actual_libraries, actual_keywords = load_train_method(method, libraries, num_of_keywords_after_dot=0)

            if predict_by_project == "True":
                # Store libraries and keywords for each project into a dictionary
                if method.repo not in test_domains_libraries.keys():
                    test_domains_libraries[method.repo] = actual_libraries
                    test_domains_keywords[method.repo] = actual_keywords
                else:
                    test_domains_libraries[method.repo] = list(
                        set(test_domains_libraries[method.repo] + actual_libraries))
                    test_domains_keywords[method.repo] = list(
                        set(test_domains_keywords[method.repo] + actual_keywords))
            else:
                test_domains_libraries[method.name] = actual_libraries
                test_domains_keywords[method.name] = actual_keywords

    counter = counter + 1

print(len(train_domains))
print(len(test_domains_libraries))
