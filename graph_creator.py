import tokenize_files
import os
import re
import networkx as nx
import matplotlib.pyplot as plt

#Returns all the paths of the files in a directory
#Starts from the files in the directory and then from the directories top bottom 
def get_all_paths(path):

    file_paths = []
    for r, d, f in os.walk(path):
        for file in f:
            if '.py' in file:
                file_paths.append(os.path.join(r, file))
    return file_paths

if __name__ == "__main__": 

    dir_path=os.getcwd()+'\keras\\tests'

    file_paths= get_all_paths(dir_path)
    libraries=[]
    keywords=[]
    G=nx.Graph()

    for file_path in file_paths:

        path_libraries,path_keywords=tokenize_files.get_libs_and_keywords(file_path)

        index=file_paths.index(file_path)+1
        print("Path Tokenazation {0} of {1} has finished".format(str(index), str(len(file_paths))),str(index/len(file_paths)*100),"%","of 100%")

        libraries = list(set(libraries + path_libraries))
        keywords = list(set(keywords + path_keywords))
       
       #Add libraries as nodes in the graph
        for string in  path_libraries:
            if string not in libraries:
                G.add_node(string)

        #Add keyword as nodes in the graph
        for string in path_keywords:
            if string not in keywords:
                G.add_node(string) 

        #Connect each library with all the keyowords
        for string in  path_libraries:
            for string2 in path_keywords:
                G.add_edge(string,string2)

    #Matches a string with 1 to 3 numbers, zero or more "." and one or 2 letters
    pattern=re.compile(r'[0-9]{1,3}.*?[a-z]{1,2}')
    for string in keywords:
        if( pattern.match(string)):
            keywords.remove(string)
        #if (string.isdigit()):
            #keywords.remove(string)
        if ( len(string)==1):
             keywords.remove(string)

    for key in keywords:
        #Remove strings like y1, 1y etc.
        if (len(key)==2):
           if any(c.isdigit() for c in key):
                keywords.remove(key)
        #Remove string like y_1, y_b etc.
        if (len(key)==3 and key[1]=='_'):
            keywords.remove(key)  

    print("Number of unique libraries:",len(libraries))
    print("Libraries listed alphabetically:")
    libraries.sort()
    print(libraries)

    print("Number of unique keywords:",len(keywords))
    keywords.sort()
    print("Keywords listed alphabetically:")
    print(keywords)

    #print(G.nodes(data=True))
    #print("Nodes of graph: ")
    #print(G.nodes())
    #print("Edges of graph: ")
    #print(G.edges())

    #nx.draw(G,with_labels=True)

    #plt.show()

