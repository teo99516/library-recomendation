from graph_creator import  get_all_paths
from graph_creator import  add_values_to_graph
from tokenize_files import remove_unwanted_words
from tokenize_files import parse_lines_with_libraries
import re
import networkx as nx
import spacy
from spacy.tokens import Doc
from spacy.lang.en import English
import matplotlib.pyplot as plt

nlp_eng = English()
# nltk.download('averaged_perceptron_tagger')
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

def line_by_line_graph(file_paths):

    keywords_graph = nx.Graph()
    libraries=[]

    for file_path in file_paths:
        python_file = open(file_path, 'r')
        print("file_path: ",file_path)
        libraries_dict = {}
        for code_line in python_file:
            if ('import' or 'from' or 'as') in code_line:
                # Removes unwanted characters by replacing them with spaces
                line_code_as_text = re.sub(r"\(|\)|\:|\[|\]|\"|\'|\||\\|\{|\}|\=|\+|\-|\*|\/|\%|\<|\>|\@|\!|\`|\,",
                                           " ", code_line)
                # Split the text line by line
                splitted_code_line = line_code_as_text.split()

                # Get the library's full name and store it to a dictionary
                libraries_full_name, libraries_dict = parse_lines_with_libraries(splitted_code_line, libraries_dict)
                # Add libraries in this file
                # e.g. when keras.layers.Dense is imported -> keras, keras.layers, keras.layers.Dense are added
                for full_library in libraries_full_name:
                    libraries_to_add = full_library.split('.')
                    temp_library = libraries_to_add[0]
                    libraries.append(temp_library)
                    for library in libraries_to_add[1:]:
                        temp_library = temp_library + '.' + library
                        libraries.append(temp_library)
            else:
                line_code_as_text = re.sub(r"\(|\)|\:|\.|\[|\]|\"|\'|\||\\|\{|\}|\=|\+|\-|\*|\/|\%|\.|\,|\<|\>|\_|\@|\!|\`|\#",
                                               " ", code_line)
                splitted_code_line = line_code_as_text.split()
                # Delete unwanted keywords
                splitted_code_line=[x.lower() for x in splitted_code_line]
                keywords_temp=[]
                for keyword in splitted_code_line:
                    if not (keyword.isdigit()):
                        if any(c.isalpha() for c in keyword) and len(keyword) > 1:
                            if not nlp.vocab[keyword].is_stop:
                                keywords_temp.append(keyword)

                keywords_temp = remove_unwanted_words(keywords_temp)
                if len(keywords_temp)>=2:
                    edges_to_add=[]
                    for i in range(0,len(keywords_temp)):
                        for j in range(i+1, len(keywords_temp)):
                            if keywords_temp[i] != keywords_temp[j]:
                                edges_to_add.append([keywords_temp[i],keywords_temp[j]])
                    # Upgrade the values of the graph
                    keywords_graph = add_values_to_graph(keywords_temp, keywords_temp, keywords_graph)

    keywords=[keyword for keyword in keywords_graph.nodes() if keyword not in library]
    libraries=list(set(libraries))
    print("Number of unique libraries: ", len(libraries))
    print("Libraries listed alphabetically:")
    libraries.sort()
    print(libraries)

    print("Number of unique keywords: ", len(keywords))
    print("Keywords listed alphabetically:")
    keywords.sort()
    print(keywords)

    #nx.draw_networkx(keywords_graph,pos=nx.spring_layout(keywords_graph))
    #plt.show()

    return libraries, keywords, keywords_graph

if __name__ == "__main__":

    file_paths = get_all_paths('keras\\tests')
    libraries, keywords, keywords_graph=line_by_line_graph(file_paths)