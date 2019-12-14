import os
import re
import nltk
import spacy
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from spacy.tokens import Doc
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS

nlp_eng = English()
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nlp=spacy.load('en_core_web_sm',disable=['parser', 'ner'])

def get_libs_and_keywords(path):
  
    file=open(path,'r')
    libraries=[]
    keywords=[]
    lib_dict={}

    for line in file:

        line_text = re.sub(r"\(|\)|\:|\[|\]|\"|\'|\||\\|\{|\}|\=|\+|\-|\*|\/|\%|\,|\<|\>", " ", line)
        line_text = line_text.split()
        
        if ('import' or 'from' or 'as') in line_text:
            
            #Store the values of the libraries that was imported as something else in a dictionary 
            if 'as' in line_text:
                lib_dict[line_text[line_text.index('as')+1].lower()]=line_text[line_text.index('import')+1]
                #Remove string after 'as', keep only the imported library
                line_text.pop(line_text.index('as')+1)

            for string in line_text:  
                if string!='import' and string!='from' and string!='as'and (string.lower() not in libraries):                   
                    libraries.append(string.lower())

        else:
            for string in line_text:

                string=string.lower()
                #Removes unwanted characters from the keyword strings
                if string.endswith('.'):
                    string=string.translate({ord('.'): None})
                if string.startswith('.'):
                    string=string.translate({ord('.'): None})
                if(string.endswith("_")):
                    string=string.translate({ord('_'): None})
                if(string.startswith("_")):
                    string=string.translate({ord('_'): None})
                if(string.endswith("`")):
                    string=string.translate({ord('`'): None})
                if(string.startswith("`")):
                    string=string.translate({ord('`'): None})
                if(string.endswith("!")):
                    string=string.translate({ord('!'): None})

                #Replace the libraries that was imported as a different name with the real library name
                #(key-dot should exist in the string in order for the found key to be the actual library and not just a character in a string)
                for key in lib_dict.keys():
                    if (key+'.')in string :                      
                        string=string.replace( key, lib_dict[key] )
                    
                #Check if the string contains characters and if it already exists(ignore case)
                if not(string.isdigit()):
                    if any(c.isalpha() for c in string) and (string not in keywords) and (len(string)>1):
                            if not(nlp.vocab[string].is_stop):               
                                keywords.append(string)
   
    #Spacy tokens lemmatization
    doc_keywords = Doc(nlp.vocab, words=keywords)
    keywords= [word.lemma_ for word in doc_keywords]

    return libraries, keywords

if __name__ == "__main__": 

    path=os.getcwd()+'\keras\\tests'

    #Returns all the paths of the files in a directory
    #Starts from the files in the directory and then from the directories top bottom 
    file_paths = []
    for r, d, f in os.walk(path):
        for file in f:
            if '.py' in file:
                file_paths.append(os.path.join(r, file))

    library,keywords=get_libs_and_keywords(file_paths[7])

    library.sort()
    print(library)
    #keywords.sort()
    print(keywords)


    
 

    