import os
import re
import matplotlib.pyplot as plt


def getLibsandKeywords(path):

    file=open(path,'r')
    #file_as_text=file.read()

    library=[]
    keywords=[]
    for line in file:
        line_text = re.sub(r"\(|\)|\:|\[|\]|\"|\'|\||\\|\{|\}|\=|\+|\-|\*|\/|\%|\,|\<|\>", " ", line)
        line_text = line_text.split()
        if ('import' or 'from' or 'as') in line_text:
            for string in line_text:
                if string!='import' and string!='from' and string!='as'and (string not in library):
                    library.append(string)
        else:
            for string in line_text:
                #Check if the string contains characters and if it already exists
                if any(c.isalpha() for c in string) and (string not in keywords):
                    keywords.append(string)

    return library, keywords


if __name__ == "__main__": 

    path=os.getcwd()+'\keras\\tests'

    #Returns all the paths of the files in a directory
    #Starts from the files in the directory and then from the directories top bottom 
    file_paths = []
    for r, d, f in os.walk(path):
        for file in f:
            if '.py' in file:
                file_paths.append(os.path.join(r, file))

    library,keywords=getLibsandKeywords(file_paths[0])
    print(library)
    print(keywords)


    
 

    