import os
import re
import nltk
import spacy
from spacy.tokens import Doc
from spacy.lang.en import English

nlp_eng = English()
nltk.download('averaged_perceptron_tagger')
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])


def get_libs_and_keywords(path):

    python_file = open(path, 'r')
    libraries = []
    keywords = []
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
            libraries = libraries + libraries_full_name

        else:
            line_code_as_text = re.sub(r"\(|\)|\:|\.|\[|\]|\"|\'|\||\\|\{|\}|\=|\+|\-|\*|\/|\%|\.|\,|\<|\>|\_|\@|\!|\`",
                                       " ", code_line)
            splitted_code_line = line_code_as_text.split()

            for keyword in splitted_code_line:

                keyword = keyword.lower()
                # Replace the libraries that was imported as a different name with the real library name
                for key in libraries_dict.keys():
                    if key == keyword:
                        keyword = keyword.replace(key, libraries_dict[key])

                # Check if the string contains characters and if it already exists(ignore case)
                if not (keyword.isdigit()):
                    if any(c.isalpha() for c in keyword) and (keyword not in keywords) and len(keyword) > 1 and (
                            keyword not in libraries):
                        if not nlp.vocab[keyword].is_stop:
                            keywords.append(keyword)

    # Spacy tokens lemmatization
    # doc_keywords = Doc(nlp.vocab, words=keywords)
    # keywords= [word.lemma_ for word in doc_keywords]
    return libraries, keywords


# Store the values of the libraries, in a dictionary, according to the way that they were imported
# Returns the libraries full name
# e.g from keras import backend as k
#       --> keras.backend is stored and pair value(k,keras.backend) is stored at the dict
# e.g from keras.models import Model, Sequential
#       --> keras.models.Model, keras.models.Sequential is stored
def parse_lines_with_libraries(splitted_line, lib_dict):
    libraries_full_names = []
    # Check what's the type of library import
    if 'as' in splitted_line:
        if 'from' in splitted_line:
            library_original_name = splitted_line[splitted_line.index('from') + 1] + '.' + \
                                    splitted_line[splitted_line.index('import') + 1]
            lib_dict[splitted_line[splitted_line.index('as') + 1].lower()] = library_original_name
            libraries_full_names.append(library_original_name)

        else:
            library_original_name = splitted_line[splitted_line.index('import') + 1]
            lib_dict[splitted_line[splitted_line.index('as') + 1].lower()] = library_original_name
            libraries_full_names.append(library_original_name)
    elif 'from' in splitted_line:
        # When import from a library we can import multiple libraries
        # If multiple libraries were imported, we should store them all
        libraries_after_import = splitted_line[splitted_line.index('import') + 1:len(splitted_line)]
        for library in libraries_after_import:
            library_original_name = splitted_line[splitted_line.index('from') + 1] + '.' + library
            libraries_full_names.append(library_original_name)
            lib_dict[library] = library_original_name
    elif 'import' in splitted_line:
        library_original_name = splitted_line[splitted_line.index('import') + 1]
        libraries_full_names.append(library_original_name)

    return libraries_full_names, lib_dict


# THIS EXISTS JUST FOR TESTING
# SHOULD BE DELETED
if __name__ == "__main__":

    path = os.getcwd() + '\keras\\tests'
    # Returns all the paths of the files in a directory
    # Starts from the files in the directory and then from the directories top bottom
    file_paths = []
    for r, d, f in os.walk(path):
        for file in f:
            if '.py' in file:
                file_paths.append(os.path.join(r, file))

    library, keywords = get_libs_and_keywords(file_paths[8])

    # library.sort()
    print(library)
    # keywords.sort()
    print(keywords)
