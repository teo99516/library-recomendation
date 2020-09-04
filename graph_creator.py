import tokenize_files
import os
import re
import networkx as nx
import matplotlib.pyplot as plt


# Returns all the paths of the files in a directory
# Starts from the files in the directory and then from the directories top bottom
def get_all_paths(dir_path):
    file_paths = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if '.py' in file:
                file_paths.append(os.path.join(root, file))

    return file_paths


# Create a graph from a path with multiple files
def create_graph(file_paths):
    libraries = []
    keywords = []
    lib_key_graph = nx.Graph()

    train_domains_libraries = {}
    train_domains_keywords = {}
    for file_path in file_paths:
        # Get the path's libraries and keyword for the specific file in the path
        path_libraries, path_keywords = tokenize_files.get_libs_and_keywords(file_path)

        index = file_paths.index(file_path) + 1
        print("Path Tokenizing {0} of {1} has finished".format(str(index), str(len(file_paths))),
              str(index / len(file_paths) * 100), "%", "of 100%")

        # Upgrade the weight if an edge exists on the graph or add the edge if it does not
        lib_key_graph = add_values_to_graph(path_libraries, path_keywords, lib_key_graph)

        train_domains_libraries[file_path] = path_libraries
        train_domains_keywords[file_path] = path_keywords

        libraries = list(set(libraries + path_libraries))
        keywords = list(set(keywords + path_keywords))

    print("Number of unique libraries: ", len(libraries))
    print("Libraries listed alphabetically:")
    libraries.sort()
    print(libraries)

    print("Number of unique keywords: ", len(keywords))
    print("Keywords listed alphabetically:")
    keywords.sort()
    print(keywords)

    print("Number of Nodes in the graph: ", len(lib_key_graph.nodes()))
    print("Number of Edges in the graph: ", len(lib_key_graph.edges()))

    return libraries, keywords, lib_key_graph, train_domains_libraries, train_domains_keywords


def add_values_to_graph(path_libraries, path_keywords, lib_key_graph):
    # Check if an edge exits in the graph. If so, then increase weight by one
    lib_key_graph.add_weighted_edges_from(
        [('lib:'+ library, keyword, int(lib_key_graph.get_edge_data('lib:'+ library, keyword)['weight']) + 1)
         for library in path_libraries for keyword in path_keywords
         if lib_key_graph.has_edge('lib:'+ library, keyword)])

    # Connect each library with all the keywords(nodes are creates automatically if they don't' exist on the graph)
    lib_key_graph.add_weighted_edges_from(
        [('lib:'+ library, keyword, 1) for library in path_libraries for keyword in path_keywords
         if not (lib_key_graph.has_edge('lib:'+ library, keyword))])

    return lib_key_graph


def plot_graph(libraries, keywords, graph, total_graph=True):
    max_subgraph = nx.Graph()
    if not (total_graph):
        # Create a subgraph with only edges with large values
        for library in libraries:
            for keyword in keywords:
                if G.has_edge(library, keyword):
                    label = G.get_edge_data(library, keyword)
                    if int(label['weight']) > 20:
                        max_subgraph.add_edge(library, keyword, weight=int(label['weight']))
    else:
        max_subgraph = graph
    # Get the weights for each edge
    labels = nx.get_edge_attributes(max_subgraph, 'weight')
    # Draw the subgraph network(with large weight values) using labels for the nodes
    nx.draw_networkx(max_subgraph, pos=nx.spring_layout(max_subgraph), with_labels=True)
    # Draw the labels of the edges
    edge_labels = nx.draw_networkx_edge_labels(max_subgraph, pos=nx.spring_layout(max_subgraph), edge_labels=labels)

    plt.show()

