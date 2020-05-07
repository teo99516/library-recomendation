from tokenize_files import get_libs_and_keywords_file
from tokenize_files import remove_unwanted_words
from heapq import nlargest
import time
from graph_creator import add_values_to_graph
import networkx as nx


def pretty(text):
    return text.replace(' DCNL ', '\n ').replace(' DCSP ', ' ')


class Method:
    def __init__(self, description, declaration, meta, body):
        self.description = pretty(description[1:-1]).split('\n')[0]
        self.declaration = declaration[declaration.find("def") + 4:-1]
        self.meta = meta[:-1]
        self.body = pretty(body[:-1])
        self.domain = "/".join(self.meta.split('/')[:2])
        self.commit = "/".join(self.meta.split('/')[:3])
        self.repo = self.commit.split('_')[0]
        self.file = self.meta.split('/')[-1].split('.')[0]
        self.name = self.declaration.split('(')[0].rstrip()

    def __str__(self):
        return self.name  # self.description+"\n"+self.declaration+"\n"+self.body


class Loader:
    def __init__(self, data, limit):
        """
            Data source: https://github.com/EdinburghNLP/code-docstring-corpus
            @article{barone2017parallel,
              title={A parallel corpus of Python functions and documentation strings for automated code documentation and code generation},
              author={Barone, Antonio Valerio Miceli and Sennrich, Rico},
              journal={arXiv preprint arXiv:1707.02275},
              year={2017}
            }
        """
        self.data = data
        self.limit = limit

    def __iter__(self):
        self.desc_file = open(self.data + "_desc", "r", encoding='utf8', errors='ignore')
        self.decl_file = open(self.data + "_decl", "r", encoding='utf8', errors='ignore')
        self.meta_file = open(self.data + "_meta", "r", encoding='utf8', errors='ignore')
        self.bodies_file = open(self.data + "_bodies", "r", encoding='utf8', errors='ignore')
        return self

    def __next__(self):
        self.limit -= 1
        if self.limit == -1:
            self.desc_file.close()
            self.decl_file.close()
            self.meta_file.close()
            self.bodies_file.close()
            raise StopIteration
        return Method(next(self.desc_file), next(self.decl_file), next(self.meta_file), next(self.bodies_file))


def load(data="dataset/parallel/parallel", limit=300000):
    return iter(Loader(data, limit))


# Split keywords with "." into libraries bases on number of keywords after dot used
def split_libraries(keyword, actual_libraries, num_of_keywords_after_dot=1):
    libraries_splittted = keyword.split('.')
    libraries_to_add = [libraries_splittted[0]]
    previous_library = libraries_splittted[0]
    for key in libraries_splittted[1:]:
        libraries_to_add.append(previous_library + '.' + key)
        previous_library = previous_library + '.' + key
    # Stores the library and the libraries inside (e.g. os, os.paths etc.)
    if num_of_keywords_after_dot == 0:
        if len(libraries_to_add[0]) > 1:
            actual_libraries = actual_libraries + [libraries_to_add[0]]
    elif num_of_keywords_after_dot == 1:
        if len(libraries_to_add[0]) > 1 and len(libraries_to_add[1]) > 1:
            actual_libraries = actual_libraries + libraries_to_add[0:2]
    elif num_of_keywords_after_dot == 2:
        if len(libraries_to_add[0]) > 1 and len(libraries_to_add[1]) > 1:
            actual_libraries = actual_libraries + libraries_to_add[0:3]
    else:
        if len(libraries_to_add[0]) > 1 and len(libraries_to_add[1]) > 1:
            actual_libraries = actual_libraries + libraries_to_add[0:]

    return actual_libraries


# Calculate the actual libraries and the actual keywords of the tokenized keywords
def get_libraries_and_keywords(keywords_to_test, libraries,num_of_keywords_after_dot):
    actual_libraries = []
    actual_keywords = []
    for keyword in keywords_to_test:
        # If "." exists its a library probably, if not, its a keyword
        if "." in keyword:
            # If first or the last character is '.', its not a library (its an object probably)
            if not keyword.startswith('.') and not (keyword.endswith('.')):
                if "append" not in keyword:
                    actual_libraries = split_libraries(keyword, actual_libraries, num_of_keywords_after_dot)
            else:
                libraries_splittted = keyword.split('.')
                for key in libraries_splittted:
                    if key not in libraries and keyword not in actual_libraries and len(key) > 1:
                        actual_keywords.append(key)
        else:
            if keyword not in libraries and keyword not in actual_libraries and len(keyword) > 1:
                actual_keywords.append(keyword)

    return actual_libraries, actual_keywords


def load_dataset(number_of_methods=10000, num_of_keywords_after_dot=1):
    methods = load(limit=number_of_methods)
    counter = 1

    train_domains_keywords = {}
    train_domains_bodies = {}
    train_domains_libraries = {}

    test_domains_keywords = {}
    test_domains_libraries = {}

    libraries = []
    keywords = []
    times_used = {}
    lib_key_graph = nx.Graph()
    for method in methods:
        if counter <= 0.8 * number_of_methods:
            print(method.domain)
            #print(method.body)
            libraries_to_test, keywords_to_test = get_libs_and_keywords_file(method.body, double_keywords_held=True,
                                                                             dot_break=False)
            # Split keywords and libraries from dots
            actual_libraries, actual_keywords = get_libraries_and_keywords(keywords_to_test, libraries, num_of_keywords_after_dot)
            actual_libraries = list(set(actual_libraries + libraries_to_test))
            actual_keywords = remove_unwanted_words(actual_keywords)
            actual_libraries = remove_unwanted_words(actual_libraries)

            # Add pairs of libraries and keywords into the graph of co-occurring
            lib_key_graph = add_values_to_graph(actual_libraries, actual_keywords, lib_key_graph)
            '''
            # Store the number of times each library is used
            for library in actual_libraries:
                if library in times_used.keys():
                    times_used[library] = times_used[library] + 1
                else:
                    times_used[library] = 1

            # Store libraries and keywords for each project into a dictionary
            if method.domain not in train_domains_libraries.keys():
                train_domains_libraries[method.domain] = actual_libraries
                train_domains_keywords[method.domain] = actual_keywords
            else:
                train_domains_libraries[method.domain] = list(
                    set(train_domains_libraries[method.domain] + actual_libraries))
                train_domains_keywords[method.domain] = list(
                    set(train_domains_keywords[method.domain] + actual_keywords))

            # Store libraries bodies for each project into a dictionary
            if method.domain not in train_domains_bodies.keys():
                train_domains_bodies[method.domain] = method.body
            else:
                train_domains_bodies[method.domain] = train_domains_bodies[method.domain] + method.body
            '''
            #print(actual_libraries)
            #print(actual_keywords)
            libraries = libraries + actual_libraries
            keywords = keywords + actual_keywords
            print("Progress: ", counter / (number_of_methods * 0.8) * 100, "%")
            counter = counter + 1
        else:
            if method.domain not in train_domains_libraries.keys():
                print(method.domain)
                # print(method.body)
                libraries_to_test, keywords_to_test = get_libs_and_keywords_file(method.body, double_keywords_held=True,
                                                                                 dot_break=False)
                # Split keywords and libraries
                actual_libraries, actual_keywords = get_libraries_and_keywords(keywords_to_test, libraries,
                                                                               num_of_keywords_after_dot)
                actual_libraries = list(set(actual_libraries + libraries_to_test))
                actual_keywords = remove_unwanted_words(actual_keywords)
                actual_libraries = remove_unwanted_words(actual_libraries)

                # Store libraries and keywords for each project into a dictionary
                if method.domain not in test_domains_libraries.keys():
                    test_domains_libraries[method.domain] = actual_libraries
                    test_domains_keywords[method.domain] = actual_keywords
                else:
                    test_domains_libraries[method.domain] = list(
                        set(test_domains_libraries[method.domain] + actual_libraries))
                    test_domains_keywords[method.domain] = list(
                        set(test_domains_keywords[method.domain] + actual_keywords))

            counter = counter + 1
    '''keywords_print = train_domains_keywords["github/tensorflow"]
    keywords_print.sort()
    libraries_print = train_domains_libraries["github/tensorflow"]
    libraries_print.sort()
    print("Keywords")
    print(keywords_print)
    print("Libraries")
    print(libraries_print)
    # print(domains_bodies['github/tensorflow'])
    print("Number of times each library was used in all the projects: ")
    top_libraries = nlargest(20, times_used, key=times_used.get)
    print(top_libraries)'''
    print("Number of Nodes in the graph: ", len(lib_key_graph.nodes()))
    print("Number of Edges in the graph: ", len(lib_key_graph.edges()))
    libraries = list(set(libraries + actual_libraries))
    keywords = list(set(keywords + actual_keywords))
    print(len(libraries))
    print(len(keywords))

    for library in libraries:
        if library not in lib_key_graph.nodes():
            libraries.remove(library)
    for keyword in keywords:
        if keyword not in lib_key_graph.nodes():
            keywords.remove(keyword)

    return libraries, keywords, lib_key_graph, test_domains_libraries, test_domains_keywords


if __name__ == "__main__":
    libraries, keywords, lib_key_graph, test_domains_libraries, test_domains_keywords \
        = load_dataset(number_of_methods=10000, num_of_keywords_after_dot=1)
    keywords.sort()
    to_print = keywords
    print(to_print)
    max_subgraph = nx.Graph()
    for library in libraries:
        for keyword in keywords:
            if lib_key_graph.has_edge(library, keyword):
                label = lib_key_graph.get_edge_data(library, keyword)
                if int(label['weight']) > 20:
                    max_subgraph.add_edge(library, keyword, weight=int(label['weight']))

    print(max_subgraph.edges.data('weight', default=1))
