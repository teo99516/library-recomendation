import networkx as nx
import numpy as np
from library_predict import calculate_similarity
from library_predict import load_model_data
from load import get_libs_and_keywords_file
from heapq import nlargest
from load import get_queries


def test_query_with_trained_model(query):
    lib_key_graph = nx.read_gpickle("line_algo/data/lib_rec.gpickle")
    libraries, keywords, idf_dict = load_model_data()

    embeddings_first = nx.read_gpickle('line_algo\data\embedding_first-order.gpickle')
    embeddings_second = nx.read_gpickle('line_algo\data\embedding_second-order.gpickle')
    embeddings = {}
    for node in lib_key_graph.nodes():
        embeddings[node] = np.concatenate((embeddings_first[node], 0.3 * embeddings_second[node]), axis=None)

    libraries = [node[4:] for node in lib_key_graph if 'lib:' in node]
    _, query_keywords = get_libs_and_keywords_file(query, double_keywords_held=True, dot_break=False, stem_use="True")
    sim = calculate_similarity(libraries, embeddings, lib_key_graph, query_keywords, idf_dict, method='cosine',
                               idf='True')
    predicted_libraries = nlargest(10, sim, key=sim.get)

    print("Path keywords: ", query_keywords)
    print("Libraries predicted: ", predicted_libraries, "\n")
    # print("Total number of libraries", len(libraries))
    # print("Total number of keywords", len(keywords))


def test_queries():
    test_keywords, test_libraries = get_queries()
    for query_number in test_keywords.keys():
        query = ""
        for key in test_keywords[query_number]:
            query = query + " " + key
        test_query_with_trained_model(query)


if __name__ == "__main__":

    # test_queries()
    stop = False
    while not stop:
        print("Give the query: ")
        query = str(input())
        test_query_with_trained_model(query)
        print("Do you want to stop? Press Stop.")
        inp = str(input()).lower()
        if inp == "stop":
            stop = True
