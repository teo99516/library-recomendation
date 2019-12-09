import os
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import spacy
from spacy.tokens import Doc
from cube.api import Cube
from nltk.corpus import stopwords 

#cube=Cube(verbose=True)
#cube.load("en")

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nlp=spacy.load('en_core_web_sm')

stop_words = set(stopwords.words('english')) 

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

def get_libs_and_keywords(path):

    
    file=open(path,'r')
    #file_as_text=file.read()
    lemmatizer = WordNetLemmatizer()
    library=[]
    keywords=[]
    for line in file:
        line_text = re.sub(r"\(|\)|\:|\[|\]|\"|\'|\||\\|\{|\}|\=|\+|\-|\*|\/|\%|\,|\<|\>", " ", line)
        line_text = line_text.split()
        if ('import' or 'from' or 'as') in line_text:
            for string in line_text:
                #if string.endswith('.'):              
                    #string.replace('.', '')
               
                if string!='import' and string!='from' and string!='as'and (string.lower() not in library)and (len(string)>1):                   
                    library.append(string.lower())
        else:
            for string in line_text:
                
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
                #Check if the string contains characters and if it already exists(ignore case)
                if any(c.isalpha() for c in string) and (string.lower() not in keywords) and (len(string)>1):
                    if not(string.isdigit()):
                        if string not in stop_words:               
                            keywords.append(string.lower())

    
    for string in keywords:
        #Remove strings like y1 etc.
        if (len(string)==2):
           if any(c.isdigit() for c in string):
                keywords.remove(string)
        #Remove string like y_1, y_b etc.
        if (len(string)==3 and string[1]=='_'):
            keywords.remove(string)


        
            


   # new=[]
   # for word in keywords:
    #    sentences=cube(word)
     #   for sentence in sentences:
      #      for entry in sentence:
       #         new.append(str(entry.lemma))
       
    #library = [cube(word).lemma for word in library]
    #keywords= [ for word in keywords]

    doc_library = Doc(nlp.vocab, words=library)
    doc_keywords = Doc(nlp.vocab, words=keywords)
    library = [word.lemma_ for word in doc_library]
    keywords= [word.lemma_ for word in doc_keywords]

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

    library,keywords=get_libs_and_keywords(file_paths[0])
    print(library)
    print(keywords)


    
 

    