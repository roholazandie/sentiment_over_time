from os.path import isfile

import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.word2vec import Word2Vec

from sentiment_analysis.directorysentences import DirectorySentences
from sentiment_analysis.file_sentences import FileSentences


class WordEmbedding(object):

    def __init__(self, directories=[], input_file=""):
        self.directories = directories
        self.input_file = input_file

    def _load_word2vec_model(self, model_file):
        if isfile(model_file):
            try:
                model = Word2Vec.load(model_file)
            except:
                model = KeyedVectors.load_word2vec_format(model_file, binary=True)
        else:
            model = self._create_word2vec_model()
        return model

    def _create_word2vec_model(self, model_file, n_dimension=100):
        if self.directories:
            sentences = DirectorySentences(self.directories)
        else:
            sentences = FileSentences(self.input_file)
        model = Word2Vec(sentences, size=n_dimension, window=5, min_count=5, workers=4)
        model.save(model_file)

        # model = Word2Vec.load(model_file)
        # print(model.wv.most_similar("woman"))
        # print(model.wv.most_similar("man"))
        # print(model.wv.most_similar("morning"))

        return model

    def create_word2vec_embeddings(self, model_file, embeddings_file, words_list_file, words_list=None):
        if words_list is None:
            words_list = []
        model = self._load_word2vec_model(model_file)
        # vocab_size = len(model.wv.vocab)
        n_dimension = model.vector_size
        # word_embeddings = np.zeros((vocab_size, n_dimension))
        vocabs = [k for (k, v) in model.wv.vocab.items()]
        all_words = []
        word_embeddings = np.zeros((1, n_dimension))
        if words_list:
            for i, vocab in enumerate(sorted(vocabs)):
                if vocab in words_list:
                    word_vector = model[vocab]
                    word_embeddings = np.vstack((word_embeddings, word_vector))
                    all_words.append(vocab)
        else:
            for i, vocab in enumerate(sorted(vocabs)):
                word_vector = model[vocab]
                word_embeddings = np.vstack((word_embeddings, word_vector))
                all_words.append(vocab)

        word_embeddings = np.delete(word_embeddings, (0), axis=0)
        np.save(embeddings_file, word_embeddings)
        np.save(words_list_file, all_words)
        return word_embeddings

    def load_word2vec_embeddings(self, embeddings_file):
        # "../word_models/twitter_models/twitter_embedding.npy"
        word_embeddings = np.load(embeddings_file)
        return word_embeddings

    def load_google_word2vec_embeddings(self):
        '''
        In this method we try to load the word2vec model which contains a list of
        vectors for each word, if we have the file (in *.npy format) we load it
        otherwise we try to create it by iterating over GoogleNews word2vec
        and extract it based on most frequent word lists that we have
        :return:
        '''
        if isfile("../word_model/google_models/google_word2vec_embedding.npy"):
            return np.load("../word_models/google_models/google_word2vec_embedding.npy")
        else:
            model_file = "../word_models/google_models/GoogleNews-vectors-negative300.bin"
            words_list = np.load("../word_models/words_list.npy")
            words_list = [word.decode("utf-8") for word in words_list]
            google_model = KeyedVectors.load_word2vec_format(model_file, binary=True)
            vocabs = [k for (k, v) in google_model.wv.vocab.items()]
            # google_word2vec_matrix = np.zeros((len(vocabs), google_model.wv.vector_size))
            matrix = np.zeros((1, google_model.wv.vector_size))
            all_words = []
            for i, vocab in enumerate(sorted(vocabs)):
                if vocab in words_list:
                    word_feature = google_model[vocab]
                    matrix = np.vstack((matrix, word_feature))
                    all_words.append(vocab)

            matrix = np.delete(matrix, (0), axis=0)
            np.save("../word_models/google_models/google_word2vec_embedding.npy", matrix)
            np.save("../word_models/google_models/google_words_list.npy", all_words)

    def get_words_list(self):
        return np.load("../word_models/words_list.npy")


if __name__ == "__main__":
    # word_embedding = WordEmbedding(directories=["../dataset/positive_reviews", "../dataset/negative_reviews"],
    #                                model_file="../word_models/word2vec.bin"
    #                                )
    # word_embedding.load_google_word2vec_embedding()
    word_embedding = WordEmbedding(input_file="../word_models/twitter_models/all_twitter_sentences.txt")
    # word_embedding._create_word2vec_model(model_file="../word_models/twitter_models/twitter_word2vec.bin")
    # word_embedding.create_word2vec_embeddings(model_file="../word_models/twitter_models/twitter_word2vec.bin",
    #                                          embeddings_file="../word_models/twitter_models/twitter_embedding.npy",
    #                                          words_list_file="../word_models/twitter_models/twitter_words.npy")

    '''
    Creating embedding file(a file containing word vectors using Googlenews pretrained vectors
    '''
    words_list = list(np.load("../word_models/all_words.npy"))
    word_embedding.create_word2vec_embeddings(
        model_file="../word_models/google_models/GoogleNews-vectors-negative300.bin",
        embeddings_file="../word_models/google_models/google_embedding.npy",
        words_list_file="../word_models/google_models/google_words.npy",
        words_list=words_list)
