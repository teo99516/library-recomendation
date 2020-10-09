import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from spacy.lang.en import English
from nltk.stem import PorterStemmer
from math import log

nlp_eng = English()
# nltk.download('averaged_perceptron_tagger')
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])


def tf_idf(train_keywords):
    df ={}
    for keywords in train_keywords.values():
        for keyword in keywords:
            df[keyword] = df.get(keyword, 0) + 1
    log2= log(2)

    return {key:log2/log(df[key]+1) for key in df}


def tf_idf_files(file_paths):
    corpus = []
    # Create a corpus with a string of all the keywords(combined with spaces) for each file
    for file_name in file_paths:
        _, keywords = get_libs_and_keywords(file_name, double_keywords_held=True)
        temp_string = ' '
        for keyword in keywords:
            temp_string = temp_string + str(keyword) + ' '
        corpus.append(temp_string)

    vectorizer = TfidfVectorizer(min_df=0)
    X = vectorizer.fit_transform(corpus)
    idf = vectorizer.idf_
    idf_dict = dict(zip(vectorizer.get_feature_names(), idf))
    # print("idf: ",idf_dict)
    return idf_dict


def get_libs_and_keywords_file(python_file, double_keywords_held=False, dot_break=True, stem_use = "True", return_libraries='True'):
    libraries = []
    keywords = []
    libraries_dict = {}
    python_file = python_file.split("\n")
    ps = PorterStemmer()

    for code_line in python_file:
        # print(code_line)

        if ('import' or 'from' or 'as') in code_line and return_libraries == 'True':

            # Get the library's full name and store it to a dictionary
            libraries_full_name, libraries_dict = parse_lines_with_libraries(code_line, libraries_dict)

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
            # Parse keywords of the line, add keywords from the line in the keywords list
            keywords = parse_keywords(code_line, libraries_dict, libraries, keywords, nlp, ps, stem_use,
                                      double_keywords_held, dot_break, )
    # Remove unwanted keywords
    keywords = remove_unwanted_words(keywords)
    libraries = remove_unwanted_words(libraries)

    # Get unique values
    libraries = list(set(libraries))
    return libraries, keywords


def get_libs_and_keywords(path, double_keywords_held=False):
    python_file = open(path, 'r')
    libraries = []
    keywords = []
    libraries_dict = {}
    ps = PorterStemmer()
    for code_line in python_file:

        if ('import' or 'from' or 'as') in code_line:

            # Get the library's full name and store it to a dictionary
            libraries_full_name, libraries_dict = parse_lines_with_libraries(code_line, libraries_dict)

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
            # Parse keywords of the line, add keywords from the line in the keywords list
            keywords = parse_keywords(code_line, libraries_dict, libraries, keywords, nlp, ps, stem_use ="False",
                                      double_keywords_held=False, dot_break=True)
    # Remove unwanted keywords
    keywords = remove_unwanted_words(keywords)
    libraries = remove_unwanted_words(libraries)

    # Get unique values
    libraries = list(set(libraries))
    return libraries, keywords


# Function for parsing a line into keywords
# Double keywords is used only in line by line graph. 
def parse_keywords(code_line, libraries_dict, libraries, keywords, nlp, ps, stem_use, double_keywords_held=False, dot_break=True):
    if dot_break:
        line_code_as_text = re.sub(r"\(|\)|\:|\[|\]|\"|\.|\'|\||\\|\{|\}|\=|\+|\-|\*|\/|\%|\,"
                                   r"|\<|\>|\_|\@|\!|\`|\?|\#|\;|\~",
                                   " ", code_line)
    else:
        line_code_as_text = re.sub(r"\(|\)|\:|\[|\]|\"|\'|\||\\|\{|\}|\=|\+|\-|\*|\/"
                                   r"|\%|\,|\<|\>|\_|\@|\!|\`|\?|\#|\;|\~",
                                   " ", code_line)
    splitted_code_line = line_code_as_text.split()

    for keyword in splitted_code_line:

        keyword = keyword.lower()
        if stem_use == "True":
            if keyword.endswith('.'):
                keyword = keyword[:-1]
            keyword = ps.stem(keyword)
        # Replace the libraries that was imported as a different name with the real library name
        if keyword in libraries_dict.keys():
            keyword = keyword.replace(keyword, libraries_dict[keyword])

        # Double keywords held for idf method
        if double_keywords_held:
            # Check if the string contains characters and if it already exists(ignore case)
            if not (keyword.isdigit()):
                if any(c.isalpha() for c in keyword) and len(keyword) > 1 and (
                        keyword not in libraries):
                    if not nlp.vocab[keyword].is_stop:
                        keywords.append(keyword)
        else:
            # Check if the string contains characters and if it already exists(ignore case)
            if not (keyword.isdigit()):
                if any(c.isalpha() for c in keyword) and (keyword not in keywords) and len(keyword) > 1 and (
                        keyword not in libraries):
                    if not nlp.vocab[keyword].is_stop:
                        keywords.append(keyword)
    #last_keyword = None
    #for keyword in list(keywords):
    #    if last_keyword is not None:
    #        keywords.append(last_keyword+ ' ' +keyword)
    #    last_keyword = keyword


    return keywords


def remove_unwanted_words(keywords):
    # Matches a string with 1 to 3 numbers, zero or more "." and one or 2 letters
    pattern = re.compile(r'[0-9]{1,3}.*?[a-z]{1,2}')
    for keyword in keywords:
        if pattern.match(keyword) or keyword.isdigit() or "www" in keyword:
            keywords.remove(keyword)
    pattern = re.compile(r'[a-z]{1,2}[0-9]{1,3}')
    for keyword in keywords:
        if pattern.match(keyword):
            keywords.remove(keyword)
    # Remove strings like y1, 1y etc.
    for keyword in keywords:
        if len(keyword) == 2:
            if any(c.isdigit() for c in keyword):
                keywords.remove(keyword)

    return keywords


# Store the values of the libraries, in a dictionary, according to the way that they were imported
# Returns the libraries full name
# e.g from keras import backend as k
#       --> keras.backend is stored and pair value(k,keras.backend) is stored at the dict
# e.g from keras.models import Model, Sequential
#       --> keras.models.Model, keras.models.Sequential is stored
def parse_lines_with_libraries(code_line, lib_dict):
    # Removes unwanted characters by replacing them with spaces
    line_code_as_text = re.sub(r"\(|\)|\:|\[|\]|\"|\'|\||\\|\{|\}|\=|\+|\-|\*|\/|\%|\<|\>|\@|\!|\`|\,",
                               " ", code_line)
    # Split the text line by line
    splitted_line = line_code_as_text.split()
    libraries_full_names = []
    # Check what's the type of library import
    if 'as' in splitted_line:
        if 'from' in splitted_line:
            if 'import' in splitted_line:
                if splitted_line.index('import') < len(splitted_line) - 1 and \
                        splitted_line.index('from') < len(splitted_line) - 1:
                    library_original_name = splitted_line[splitted_line.index('from') + 1] + '.' + \
                                            splitted_line[splitted_line.index('import') + 1]
                    lib_dict[splitted_line[splitted_line.index('as') + 1].lower()] = library_original_name
                    libraries_full_names.append(library_original_name)
        elif 'import' in splitted_line:
            if splitted_line.index('as') < len(splitted_line) - 1 and \
                    splitted_line.index('import') < len(splitted_line) - 1:
                library_original_name = splitted_line[splitted_line.index('import') + 1]
                lib_dict[splitted_line[splitted_line.index('as') + 1].lower()] = library_original_name
                libraries_full_names.append(library_original_name)
    elif 'from' in splitted_line:
        if 'import' in splitted_line:
            # When import from a library we can import multiple libraries
            # If multiple libraries were imported, we should store them all
            if splitted_line.index('import') < len(splitted_line) - 1 and \
                    splitted_line.index('from') < len(splitted_line) - 1:
                '''
                libraries_after_import = splitted_line[splitted_line.index('import') + 1:len(splitted_line)]
                for library in libraries_after_import:
                    library_original_name = splitted_line[splitted_line.index('from') + 1] + '.' + library
                    libraries_full_names.append(library_original_name)
                    lib_dict[library] = library_original_name
                '''
                library_original_name = splitted_line[splitted_line.index('from') + 1]
                libraries_full_names.append(library_original_name)
    elif 'import' in splitted_line:

        if splitted_line.index('import') < len(splitted_line) - 1:
            library_original_name = splitted_line[splitted_line.index('import') + 1]
            libraries_full_names.append(library_original_name)

    return libraries_full_names, lib_dict
