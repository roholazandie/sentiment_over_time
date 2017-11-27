import string
from nltk.corpus import stopwords
from gensim import corpora
#import spacy


class FileLazyCorpus():
    def __init__(self, file_name):
        self.file_name = file_name
        self.printable = set(string.printable)
        self._initialize_dictionary()
        #self.nlp = en_core_web_sm.load()


    def save_dictionary(self, file_name):
        self.dictionary.save(file_name)


    def _initialize_dictionary(self, ):
        words = [[filter(lambda x: x in self.printable, word) for word in line.lower().split()] for line in open(self.file_name)]
        words = [[''.join(ch for ch in word if ch.isalnum() or ch == ' ') for word in doc] for doc in words]
        self.dictionary = corpora.Dictionary(words)
        once_ids = [tokenid for tokenid, docfreq in self.dictionary.dfs.items() if docfreq == 1]
        self.dictionary.filter_tokens(once_ids)
        self.dictionary.compactify()


    def __iter__(self):
        with open(self.file_name) as file_reader:
            for document in file_reader:
                word_list = document.split()
                filtered_words = [word for word in word_list if word not in stopwords.words('english')]

                # en_abstract = self.nlp(document)
                # words = [token.lemma_ for token in en_abstract if not (token.pos_ == "NUM"
                #                                                        or token.pos_ == "SYM"
                #                                                        or token.pos_ == "PUNCT"
                #                                                        or token.pos_ == "CCONJ"
                #                                                        or token.is_stop
                #                                                        or token.like_num
                #                                                        or token.is_space
                #                                                        or token.is_punct)]
                yield self.dictionary.doc2bow(filtered_words)



class ListLazyCorpus(object):
    def __init__(self, sentence_list):
        self.sentence_list = sentence_list
        self.printable = set(string.printable)
        self._initialize_dictionary()
        #self.nlp = en_core_web_sm.load()


    def save_dictionary(self, file_name):
        self.dictionary.save(file_name)


    def _initialize_dictionary(self, ):
        words = [[filter(lambda x: x in self.printable, word) for word in line.lower().split()] for line in self.sentence_list]
        words = [[''.join(ch for ch in word if ch.isalnum() or ch == ' ') for word in doc] for doc in words]
        self.dictionary = corpora.Dictionary(words)
        once_ids = [tokenid for tokenid, docfreq in self.dictionary.dfs.items() if docfreq == 1]
        self.dictionary.filter_tokens(once_ids)
        self.dictionary.compactify()


    def __iter__(self):
        for document in self.sentence_list:
            word_list = document.split()
            filtered_words = [word for word in word_list if word not in stopwords.words('english')]

            # en_abstract = self.nlp(document)
            # words = [token.lemma_ for token in en_abstract if not (token.pos_ == "NUM"
            #                                                        or token.pos_ == "SYM"
            #                                                        or token.pos_ == "PUNCT"
            #                                                        or token.pos_ == "CCONJ"
            #                                                        or token.is_stop
            #                                                        or token.like_num
            #                                                        or token.is_space
            #                                                        or token.is_punct)]
            yield self.dictionary.doc2bow(filtered_words)