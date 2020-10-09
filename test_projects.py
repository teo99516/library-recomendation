import networkx as nx
import numpy as np
from library_predict import calculate_similarity
from library_predict import load_model_data
from library_predict import calculate_hit_rate
from library_predict import caclulate_function_similarity
from heapq import nlargest
import relation_model
import csv
from sklearn.metrics import roc_auc_score
from sklearn.metrics import ndcg_score
from load import load_dataset


def test_experiment_projects_with_embeddings():
    libraries, keywords, lib_key_graph, test_domains_libraries, test_domains_keywords, train_domains_libraries, \
    train_domains_keywords = load_dataset(number_of_methods=100000, num_of_keywords_after_dot=0)

    libraries, keywords, idf_dict = load_model_data()
    have_embeddings = "True"
    random_predict = "False"
    similarity_methods = ['cosine']
    proximities = ['both']
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
                for node in lib_key_graph.nodes():
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
                                                                scaler, model, idf_dict, idf=idf_use)
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
                results.writerow([sum(hit_rate) / len(hit_rate), sum(auc) / len(auc), sum(ndcg) / len(ndcg),
                                  len(libraries_predicted_list) / len(libraries) * 100])
                results.writerow([np.std(hit_rate), np.std(auc), np.std(ndcg)])
                coverage.append(len(libraries_predicted_list) / len(libraries))
            # results.writerow([sum(hit_rate) / len(hit_rate)*100, sum(auc) / len(auc)*100, sum(ndcg) / len(ndcg)*100,
            #                  sum(coverage)/len(coverage)*100])


