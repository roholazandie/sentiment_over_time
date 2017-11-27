from os.path import isfile

import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.word2vec import Word2Vec

from sentiment_analysis.sentences import Sentences


class WordEmbedding(object):

    def __init__(self, directories, model_file, dimension=100):
        self.n_dimension = dimension
        self.directories = directories
        self.model_file = model_file
        #self._create_word2vec_model()
        #self.model = self._create_word2vec_model()


    def _create_word2vec_model(self):
        if isfile("word_models/word2vec.bin"):
            model = Word2Vec.load(self.model_file)
        else:
            sentences = Sentences(self.directories)
            model = Word2Vec(sentences, size=self.n_dimension, window=5, min_count=5, workers=4)
            model.save(self.model_file)

        return model


    def create_word2vec_embeddings(self):
        vocab_size = len(self.model.wv.vocab)
        word_embeddings = np.zeros((vocab_size, self.n_dimension))


        for i, (k, v) in enumerate(self.model.wv.vocab.items()):
            word_embeddings[i] = self.model[k]

        np.save("word_models/word2vec_embeddings.npy", word_embeddings)
        return word_embeddings


    def load_word2vec_embeddings(self):
        #word_embeddings = np.load("word_models/word2vec_embeddings.npy")
        word_embeddings = np.load("word_models/word2vec_google_matrix.npy")
        return word_embeddings


    def load_google_word2vec_embedding(self):
        '''
        In this method we try to load the word2vec model which contains a list of
        vectors for each word, if we have the file (in *.npy format) we load it
        otherwise we try to create it by iterating over GoogleNews word2vec
        and extract it based on most frequent word lists that we have
        :return:
        '''
        if isfile("word_model/word2vec_google.npy"):
            return np.load("word_models/word2vec_google.npy")
        else:
            model_file = "word_models/GoogleNews-vectors-negative300.bin"
            words_list = np.load("word_models/words_list.npy")
            words_list = [word.decode("utf-8") for word in words_list]
            google_model = KeyedVectors.load_word2vec_format(model_file, binary=True)
            vocabs = [k for (k, v) in google_model.wv.vocab.items()]
            #google_word2vec_matrix = np.zeros((len(vocabs), google_model.wv.vector_size))
            matrix = np.zeros((1, google_model.wv.vector_size))
            all_words = []
            for i, vocab in enumerate(sorted(vocabs)):
                if vocab in words_list:
                    word_feature = google_model[vocab]
                    matrix = np.vstack((matrix, word_feature))
                    all_words.append(vocab)

            matrix = np.delete(matrix, (0), axis=0)
            np.save("word_models/word2vec_google_matrix.npy", matrix)
            np.save("word_models/all_words.npy", all_words)

    def get_words_list(self):
        # if isfile("word_models/word2vec.bin"):
        #     self.model = Word2Vec.load(self.model_file)
        # else:
        #     self.model = self._create_word2vec_model()
        #
        # words_list = [k for (k, v) in self.model.wv.vocab.items()]
        # np.save("word_models/words_list.npy", words_list)
        # return words_list
        return np.load("word_models/words_list.npy")


if __name__ == "__main__":
    word_embedding = WordEmbedding(directories=["dataset/positive_reviews", "dataset/negative_reviews"],
                                   model_file="word_models/word2vec.bin"
                                   )
    #embedding = word_embedding.create_word2vec_embeddings()
    word_embedding.load_google_word2vec_embedding()