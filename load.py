from tokenize_files import get_libs_and_keywords_file
from tokenize_files import remove_unwanted_words
from heapq import nlargest
import time
from graph_creator import add_values_to_graph
import networkx as nx
import re
import requests
import random
from collections import Counter
import pickle
from tokenize_files import tf_idf

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
def split_libraries(keyword, actual_libraries, declaration_objects, method_body, num_of_keywords_after_dot=1):
    if keyword + " = " not in method_body:
        libraries_splittted = keyword.split('.')
        if libraries_splittted[0] not in declaration_objects and libraries_splittted[0] + " = " not in method_body:
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
def get_libraries_and_keywords(keywords_to_test, libraries, declaration_objects, method_body,
                               num_of_keywords_after_dot):
    actual_libraries = []
    actual_keywords = []
    for keyword in keywords_to_test:
        # If "." exists its a library probably, if not, its a keyword
        if "." in keyword:
            # If first or the last character is '.', its not a library (its an object probably)
            if not keyword.startswith('.') and not (keyword.endswith('.')):
                actual_libraries = split_libraries(keyword, actual_libraries, declaration_objects, method_body,
                                                   num_of_keywords_after_dot)
            else:
                libraries_splittted = keyword.split('.')
                for key in libraries_splittted:
                    if key not in libraries and keyword not in actual_libraries and len(key) > 1:
                        actual_keywords.append(key)
        else:
            if keyword not in libraries and keyword not in actual_libraries and len(keyword) > 1:
                actual_keywords.append(keyword)

    return actual_libraries, actual_keywords


# Load a method's keywords and libraries
def load_train_method(method, libraries, num_of_keywords_after_dot=1, description_use = "True"):

    print(method.repo)
    # print(method.description)
    # print(method.body)
    # Keep the objects in the declaration to exclude them from being chosen as libraries
    objects = re.sub(r"\(|\)|\:|\[|\]|\"|\.|\'|\||\\|\{|\}|\=|\+|\-|\*|\/|\%|\,"
                     r"|\<|\>|\_|\@|\!|\`|\?|\#|\;|\~",
                     " ", method.declaration)
    declaration_objects = objects.split()

    # Get the libraries and keywords from methods body
    libraries_to_test, keywords_to_test = get_libs_and_keywords_file(method.body,
                                                                     double_keywords_held=True,
                                                                     dot_break=False, stem_use="False")
    if description_use == "True":
        # Get the keywords for the method's description
        _, actual_keywords = get_libs_and_keywords_file(method.description, double_keywords_held=True,
                                                        dot_break=False, stem_use="True")

        # Split keywords and libraries from dots. Keep only the libraries
        actual_libraries, _ = get_libraries_and_keywords(keywords_to_test, libraries,
                                                         declaration_objects, method.body,
                                                         num_of_keywords_after_dot)
    else:

        # Split keywords and libraries from dots. Keep only the libraries
        actual_libraries, actual_keywords = get_libraries_and_keywords(keywords_to_test, libraries,
                                                                       declaration_objects, method.body,
                                                                       num_of_keywords_after_dot)

    #actual_libraries = [library for library in libraries if library + " = " not in method.body]

    actual_libraries = list(set(actual_libraries + libraries_to_test))
    actual_keywords = list(set(actual_keywords))
    actual_keywords = remove_unwanted_words(actual_keywords)
    actual_libraries = remove_unwanted_words(actual_libraries)

    return actual_libraries, actual_keywords

# Count how many times libraries and keywords was used in the training set
def count_times_used(train_domains_libraries, train_domains_keywords ):

    times_used = {}
    times_used_libs = {}
    # Count how many times a library was used in the projects
    for repo in train_domains_libraries.keys():
        for library in train_domains_libraries[repo]:
            if library in times_used_libs.keys():
                times_used_libs[library] = times_used_libs[library] + 1
            else:
                times_used_libs[library] = 1
    # Count how many times a keyword was used in the projects
    for repo in train_domains_keywords.keys():
        for keyword in train_domains_keywords[repo]:
            if keyword in times_used.keys():
                times_used[keyword] = times_used[keyword] + 1
            else:
                times_used[keyword] = 1

    return times_used_libs, times_used


def remove_libs_and_keys(libraries, keywords, lib_key_graph, test_domains_libraries, times_used, times_used_libs,
                         LIB_THRESHOLD, KEY_THRESHOLD):
    libraries = list(set(libraries))
    keywords = list(set(keywords))

    libs_to_delete = ['', 'list', 'dict', 'struct', 'self', 'str', '.', 'set', 'filename']

    # Keep only the keywords of the train set that were used more times than the key_threshold
    keywords = [keyword for keyword in keywords if KEY_THRESHOLD <= times_used.get(keyword, 0)]

    # Keep only the libraries of the train set that were used more times than the lib_threshold
    libraries = [library for library in libraries if times_used_libs.get(library, 0) >= LIB_THRESHOLD
                 and library not in libs_to_delete and len(library) > 1]

    # Keep only the keywords of the test set that were used more times than the lib_threshold
    for repo in test_domains_libraries:
        test_domains_libraries[repo] = [library for library in test_domains_libraries[repo]
                                if times_used_libs.get(library, 0) >= LIB_THRESHOLD  and library not in libs_to_delete]

    # Remove nodes from the graph that was deleted from libraries and keywords list
    nodes = list(lib_key_graph.nodes())
    for node in nodes:
        if 'lib:' in node:
            if node[4:] not in libraries:
                lib_key_graph.remove_node(node)
        else:
            if node not in keywords:
                lib_key_graph.remove_node(node)

    lib_key_graph.remove_nodes_from(list(nx.isolates(lib_key_graph)))

    return libraries, keywords, lib_key_graph


def store_data(libraries, keywords, test_domains_libraries, test_domains_keywords, train_domains_libraries,
           train_domains_keywords):

    idf_dict = {**tf_idf(train_domains_keywords), **tf_idf(train_domains_libraries)}
    print(idf_dict)

    with open('idf.data', 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(idf_dict, filehandle)
        filehandle.close()

    with open('libraries.data', 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(libraries, filehandle)
        filehandle.close()

    with open('keywords.data', 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(keywords, filehandle)
        filehandle.close()

    with open('training_libs.txt', 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(test_domains_libraries, filehandle,protocol=pickle.HIGHEST_PROTOCOL)
        filehandle.close()

    with open('training_keys.txt', 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(test_domains_keywords, filehandle, protocol=pickle.HIGHEST_PROTOCOL)
        filehandle.close()

    with open('testing_libs.txt', 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(train_domains_libraries, filehandle)
        filehandle.close()

    with open('testing_keywords.txt', 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(train_domains_keywords, filehandle )
        filehandle.close()


def load_dataset(number_of_methods=10000, num_of_keywords_after_dot=1):
    methods = load(limit=number_of_methods)

    method_list = []
    for method in methods:
        method_list.append(method)

    #method_list = method_list[int(0.5*len(method_list)):]
    dataset_length=len(method_list)
    counter = 1
    train_by_method = "True"
    predict_by_project = "True"
    train_domains_keywords = {}
    train_domains_libraries = {}
    test_domains_keywords = {}
    test_domains_libraries = {}

    libraries = []
    keywords = []

    lib_key_graph = nx.Graph()
    for method in method_list:
        if counter <= 0.8 * dataset_length:

            # Load libraries and keywords for the method
            actual_libraries, actual_keywords = load_train_method(method, libraries, num_of_keywords_after_dot= num_of_keywords_after_dot,
                                                                  description_use= "True")

            if train_by_method == "True":
                # Add pairs of libraries and keywords into the graph of co-occurring
                lib_key_graph = add_values_to_graph(actual_libraries, actual_keywords, lib_key_graph)

            # Store libraries and keywords for each project into a dictionary
            if method.repo not in train_domains_libraries.keys():
                train_domains_libraries[method.repo] = actual_libraries
                train_domains_keywords[method.repo] = actual_keywords
            else:
                train_domains_libraries[method.repo] = list(set(train_domains_libraries[method.repo] + actual_libraries))
                train_domains_keywords[method.repo] = list(set(train_domains_keywords[method.repo] + actual_keywords))

            libraries = libraries + actual_libraries
            keywords = keywords + actual_keywords
            print("Progress: ", counter / (dataset_length * 0.8) * 100, "%")
            counter = counter + 1

        else:
            '''
            if method.repo not in train_domains_libraries.keys():

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
            '''
            _,  test_domains_keywords[1] = get_libs_and_keywords_file('calling an external command', double_keywords_held=True, dot_break=False, stem_use="True") #subprocess , os, shlex
            _,  test_domains_keywords[2] = get_libs_and_keywords_file('How to sort a dictionary by value', double_keywords_held=True, dot_break=False, stem_use="True") # operator, collections , operator.itemgetter
            _,  test_domains_keywords[3] = get_libs_and_keywords_file('How to convert string represantation of list to a list', double_keywords_held=True, dot_break=False, stem_use="True") #ast, json, re, pyparsing
            _,  test_domains_keywords[4] = get_libs_and_keywords_file('How do i merge two dictionaries in a single expression', double_keywords_held=True, dot_break=False, stem_use="True") #collections.ChainMap, itertools.chain, copy
            _,  test_domains_keywords[5] = get_libs_and_keywords_file('Converting string into datetime', double_keywords_held=True, dot_break=False, stem_use="True") #datetime ,dateutil , timestring
            _,  test_domains_keywords[6] = get_libs_and_keywords_file('How to import a module with the full path', double_keywords_held=True, dot_break=False, stem_use="True") #imp, importlib.util, runpy , pkgutil
            _,  test_domains_keywords[7] = get_libs_and_keywords_file('How can you profile a python script', double_keywords_held=True, dot_break=False, stem_use="True") #cProfile
            _,  test_domains_keywords[8] = get_libs_and_keywords_file('How to get current time', double_keywords_held=True, dot_break=False, stem_use="True")   #time, datetime , pandas, numpy, arrow
            _,  test_domains_keywords[9] = get_libs_and_keywords_file('How do I concatenate two lists', double_keywords_held=True, dot_break=False, stem_use="True")  #itertools, heapq, operator
            _,  test_domains_keywords[10] = get_libs_and_keywords_file('How to count occurrences of a list item', double_keywords_held=True, dot_break=False, stem_use="True") #numpy, more_itertools, collections
            _,  test_domains_keywords[11] = get_libs_and_keywords_file('How do I download a file over HTTP', double_keywords_held=True, dot_break=False, stem_use="True") #urllib, requests, wget, pycurl
            _,  test_domains_keywords[12] = get_libs_and_keywords_file('How to use threading', double_keywords_held=True, dot_break=False, stem_use="True") #threading, multiprocessing, concurrent
            _,  test_domains_keywords[13] = get_libs_and_keywords_file('how to connect to mysql database', double_keywords_held=True, dot_break=False, stem_use="True") #MySQLdb, peewee, mysql.connector, pymysql, oursql, flask_mysqldb
            _,  test_domains_keywords[14] = get_libs_and_keywords_file('how to write json data to a file', double_keywords_held=True, dot_break=False, stem_use="True") #json, mpu.io
            _,  test_domains_keywords[15] = get_libs_and_keywords_file('How to get POSTed JSON in Flask', double_keywords_held=True, dot_break=False, stem_use="True") #flask, requests, json

            test_domains_libraries = {number:[] for number in test_domains_keywords.keys()}


    # Construct the graph if training by project was chosen
    if train_by_method == "False":
        for repo in train_domains_libraries.keys():
            lib_key_graph = add_values_to_graph(train_domains_libraries[repo], train_domains_keywords, lib_key_graph)


    number_of_projects = len(train_domains_libraries.keys())
    KEY_THRESHOLD = 0.02 * number_of_projects + 1
    LIB_THRESHOLD = 0.04 * number_of_projects + 1

    times_used_libs, times_used = count_times_used(train_domains_libraries, train_domains_keywords)

    libraries, keywords, lib_key_graph = remove_libs_and_keys(libraries, keywords, lib_key_graph,
            test_domains_libraries, times_used, times_used_libs, LIB_THRESHOLD, KEY_THRESHOLD )

    print("Number of Nodes in the graph: ", len(lib_key_graph.nodes()))
    print("Number of Edges in the graph: ", len(lib_key_graph.edges()))
    print("Libraries ", len(libraries))
    print("Keywords ", len(keywords))
    print("Total number of projects: ", number_of_projects)

    store_data(libraries, keywords, test_domains_libraries, test_domains_keywords, train_domains_libraries,
               train_domains_keywords)

    return libraries, keywords, lib_key_graph, test_domains_libraries, test_domains_keywords, train_domains_libraries, \
           train_domains_keywords

def request_github(methods, number_of_methods, item='Topic'):
    progress=0
    repos_readme = {}
    methods_checked = []
    for method in methods:
        if method.repo not in methods_checked:
            methods_checked.append(method.repo)
            print(method.repo)
            r = requests.get('https://github.com/' + method.repo[7:])
            root = html.fromstring(r.content)
            root.make_links_absolute('https://github.com/' + method.repo[7:])
            if item == "Topic":
                tagcloud = root.xpath('//a[contains(@title, "Topic:")]/text()')
                if len(tagcloud) > 0:
                    tagcloud = [tag.strip() for tag in tagcloud]
            else:
                tagcloud = root.xpath('//article[contains(@class,"entry-content")]/p/text()')
                tagcloud = [tag for tag in tagcloud if tag != " " and tag != "."]
                repos_readme[method.repo] = tagcloud[0:2]
        print("Progress ", progress / number_of_methods * 100)
        progress = progress + 1

    return repos_readme


if __name__ == "__main__":

    #libraries, keywords, lib_key_graph, test_domains_libraries, test_domains_keywords, train_domains_libraries, \
    #train_domains_keywords = load_dataset(number_of_methods=50000, num_of_keywords_after_dot=0)

    number_of_methods = 100000
    methods = load(limit=number_of_methods)
    train_domains=[]
    counter=0
    for method in methods:
        if counter <= 0.8 * number_of_methods:
            if method.repo not in train_domains:
                train_domains.append( method.repo)
            counter = counter +1

    print(len(train_domains))



    '''
    number_of_methods = 50000
    methods = load(limit=number_of_methods)
    methods_checked = []
    tag_list = []
    progress = 0
    for method in methods:
        if method.repo not in methods_checked:
            methods_checked.append(method.repo)
            print(method.repo)
            r = requests.get('https://github.com/' + method.repo[7:])
            root = html.fromstring(r.content)
            root.make_links_absolute('https://github.com/' + method.repo[7:])
            tagcloud = root.xpath('//a[contains(@title, "Topic:")]/text()')
            tagcloud = [tag.strip() for tag in tagcloud]
            print(tagcloud)
            tag_list = tag_list + tagcloud
        #print("Progress ", progress/number_of_methods*100)
        progress = progress + 1
    print(Counter(tag_list))
    '''
    # get_readme(number_of_methods=3000)
