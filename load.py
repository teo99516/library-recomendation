from tokenize_files import get_libs_and_keywords_file


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


if __name__ == "__main__":
    '''
    for method in methods:
        method.body_tokens = {token: 1.0 for token in set(splitter.split(method.body, 0))}
        for token in set(splitter.split(method.declaration)):
            method.body_tokens[token] = 1.0
        method.body_tokens = {token: method.body_tokens[token]/len(method.body_tokens)**0.5 for token in method.body_tokens}
        method.quality = similarity(method.body_tokens, {token: 1.0 for token in set(splitter.split(method.description))})/len(method.body_tokens)
        if method.domain not in domains:
            domains[method.domain] = list()
        domains[method.domain].append(method)
    '''
    number_of_methods = 100
    methods = load(limit=number_of_methods)
    domains_keywords = {}
    domains_libraries = {}
    libraries = []
    keywords = []
    counter = 1
    times_used = {}
    for method in methods:
        print(method.domain)
        print(method.body)
        libraries_to_test, keywords_to_test = get_libs_and_keywords_file(method.body, double_keywords_held=True,
                                                                         dot_break=False)
        print("libraries test", libraries_to_test)
        actual_libraries = []
        actual_keywords = []
        for keyword in keywords_to_test:
            if "." in keyword:
                # If first character is '.', its not a library (its an object probably)
                if keyword[0] != '.':
                    lib = keyword.split('.')
                    # Stores the library and the library inside (e.g. os, os.paths)
                    if len(lib[0]) > 1 and len(lib[1]) > 1:
                        actual_libraries.append(lib[0])
                        actual_libraries.append(lib[0]+'.'+lib[1])
                else:
                    lib = keyword.split('.')
                    for key in lib:
                        if key not in actual_libraries and len(key) > 1:
                            actual_keywords.append(key)
            else:
                if keyword not in actual_libraries and len(keyword) > 1:
                    actual_keywords.append(keyword)

        actual_libraries = list(set(actual_libraries + libraries_to_test))

        # Store the number of times each library is used
        for library in actual_libraries:
            if library in times_used.keys():
                times_used[library] = times_used[library]+1
            else:
                times_used[library] = 1

        # Store libraries and keywords for each project into a dictionary
        if method.domain not in domains_libraries.keys():
            domains_libraries[method.domain] = actual_libraries
            domains_keywords[method.domain] = actual_keywords
        else:
            domains_libraries[method.domain] = list(set(domains_libraries[method.domain] + actual_libraries))
            domains_keywords[method.domain] = list(set(domains_keywords[method.domain] + actual_keywords))

        libraries = list(set(libraries + actual_libraries))
        keywords = list(set(keywords + actual_keywords))
        print("Progress: ", counter / number_of_methods * 100, "%")
        counter = counter + 1

    keywords_print = domains_keywords["github/gstarnberger"]
    keywords_print.sort()
    libraries_print = domains_libraries["github/gstarnberger"]
    libraries_print.sort()
    print("Keywords")
    print(keywords_print)
    print("Libraries")
    print(libraries_print)
    print("Number of times each library was used in all the projects: ")
    print(times_used)
    # libraries.sort()
    # print(libraries)
    # keywords.sort()
    # print(keywords)
