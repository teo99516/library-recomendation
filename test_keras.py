import graph_creator
import networkx as nx
import tokenize_files
import numpy as np
from heapq import nlargest
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import ndcg_score
from graph_line_by_line import line_by_line_graph
import relation_model
import csv
from library_predict import get_embeddings, caclulate_function_similarity, calculate_similarity, \
    predict_without_embeddings, calculate_hit_rate


def test_keras():
    libraries, keywords, lib_key_graph, test_domains_libraries, test_domains_keywords, \
    train_domains_libraries, train_domains_keywords = load_keras()

    # Store graph in a file
    nx.write_gpickle(lib_key_graph, "line_algo/data/keras-graph.gpickle")
    embeddings_method = "line"
    have_embeddings = "True"
    random_predict = "False"
    similarity_methods = ['cosine', 'dot', 'function']
    proximities = ['first', 'second']
    idf_uses = ['True']

    if have_embeddings == "False":
        predict_without_embeddings(lib_key_graph, libraries, test_domains_libraries, test_domains_keywords,
                                   random_predict=random_predict)
    else:
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

                        embeddings = get_embeddings(lib_key_graph, graph_name="keras-graph", proximity_method=proximity)

                        if similarity_method == 'function':
                            # Create training set for the model of the similarity prediction
                            training_features, training_values = relation_model.create_training_set(lib_key_graph,
                                                                                                    embeddings,
                                                                                                    libraries, keywords)
                            # Train the relation model
                            scaler, model = relation_model.train_relation_model(training_features, training_values)

                        libraries_predicted_list = []

                        for domain in test_domains_libraries.keys():
                            # Check if libraries was identified in this domain
                            if len(test_domains_libraries[domain]) >= 0:  # and len(test_domains_libraries[domain])> 5:

                                path_keywords = test_domains_keywords[domain]
                                path_libraries = test_domains_libraries[domain]
                                print('Predict path: ', domain)
                                print("Number of libraries in this file: ", len(test_domains_libraries[domain]))

                                # Calculate similarity and save it in a dictionary
                                if similarity_method == "function":
                                    sim = caclulate_function_similarity(libraries, embeddings, lib_key_graph,
                                                                        path_keywords,
                                                                        scaler,
                                                                        model,
                                                                        idf_dict, idf=idf_use)
                                else:
                                    sim = calculate_similarity(libraries, embeddings, lib_key_graph, path_keywords,
                                                               idf_dict,
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
                                print("Discounted Cumulative Gain: ", ndcg_score([np.array(labels)], [np.array(conf)]),
                                      '\n')
                        libraries_predicted_list = list(set(libraries_predicted_list))
                        coverage.append(len(libraries_predicted_list) / len(libraries))
                    results.writerow([sum(hit_rate) / len(hit_rate), sum(auc) / len(auc), sum(ndcg) / len(ndcg),
                                      len(libraries_predicted_list) / len(libraries) * 100])
                # results.writerow([sum(hit_rate) / len(hit_rate), sum(auc) / len(auc), sum(ndcg) / len(ndcg),
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


def load_keras(graph_method="co-occur"):
    # Get all the paths inside the directory path provided before
    file_paths = graph_creator.get_all_paths('keras\\tests')

    # Training 80% Testing 20%
    training_set, test_set = train_test_split(file_paths, test_size=0.2, random_state=0)

    test_domains_keywords = {}
    test_domains_libraries = {}

    # Create the graph of the training set with the appropriate method
    if graph_method == "co-occur":
        libraries, keywords, lib_key_graph, train_domains_libraries, train_domains_keywords = graph_creator.create_graph(
            training_set)
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
    test_keras()
