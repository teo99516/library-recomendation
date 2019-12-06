import tokenize_files
import os
import re
import networkx as nx
import matplotlib.pyplot as plt

path=os.getcwd()+'\keras\\tests'

#Returns all the paths of the files in a directory
#Starts from the files in the directory and then from the directories top bottom 
file_paths = []
for r, d, f in os.walk(path):
    for file in f:
        if '.py' in file:
            file_paths.append(os.path.join(r, file))

library,keywords=tokenize_files.getLibsandKeywords(file_paths[0])

print(library)
print(keywords)

G=nx.Graph()
index=0
for string in library:
    G.add_node( index,library=string)
    index=index+1 

print(G.nodes(data=True))
#pos=nx.spring_layout(G)


nx.draw(G)

plt.show()

