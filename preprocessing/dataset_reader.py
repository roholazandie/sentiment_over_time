import os
import re
from os.path import isfile
from random import randint
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sentiment_analysis.word_embedding import WordEmbedding

strip_special_chars = re.compile("[^A-Za-z0-9 ]+")

'''
#twitter settings
WORD_EMBEDDING_FILE = "../word_models/twitter_models/twitter_embedding.npy"
WORDS_LIST_FILE = "../word_models/twitter_models/twitter_words.npy"
WORD_INDEX_MATRIX_FILE = "../word_models/twitter_models/twitter_word_index_matrix_maxseq=30.npy"
LABELS_FILE = "../word_models/twitter_models/twitter_labels_maxseq=30.npy"
'''

'''
WORD_EMBEDDING_FILE = "../word_models/google_models/google_embedding.npy"
WORDS_LIST_FILE = "../word_models/google_models/google_words.npy"
WORD_INDEX_MATRIX_FILE = "../word_models/google_models/google_word_index_matrix_maxseq=30.npy"
LABELS_FILE = "../word_models/google_models/google_labels_maxseq=30.npy"
'''

'''
WORD_EMBEDDING_FILE = "../word_models/movie_review_models/movie_embedding.npy"
WORDS_LIST_FILE = "../word_models/movie_review_models/movie_words.npy"
WORD_INDEX_MATRIX_FILE = "../word_models/movie_review_models/movie_word_index_matrix_maxseq=30.npy"
LABELS_FILE = "../word_models/movie_review_models/movie_labels_maxseq=30.npy"
'''

WORD_EMBEDDING_FILE = "../word_models/tweet_models/tweet_embedding.npy"
WORDS_LIST_FILE = "../word_models/tweet_models/tweet_words.npy"
WORD_INDEX_MATRIX_FILE = "../word_models/tweet_models/tweet_word_index_matrix.npy"
LABELS_FILE = "../word_models/tweet_models/tweet_labels.npy"


class SentimentDatasetReader(object):

    def __init__(self, batch_size=None, max_seq_length=None, dataset_dirs=[], input_file=""):
        self.dataset_dirs = dataset_dirs  # ["../dataset/movie_reviews/negative_reviews", "../dataset/movie_reviews/positive_reviews"]
        self.input_file = input_file
        self.word_embedding = WordEmbedding(directories=self.dataset_dirs, input_file=self.input_file)
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length

        self.n_files = self._get_n_files()
        self.word_vectors = self.load_word_vectors(WORD_EMBEDDING_FILE)
        self.words_list = self._load_word_list(WORDS_LIST_FILE)
        if isfile(WORD_INDEX_MATRIX_FILE):
            self.dataset_train_test_split(word_index_matrix_file=WORD_INDEX_MATRIX_FILE,
                                          labels_file=LABELS_FILE)

    def dataset_train_test_split(self, word_index_matrix_file, labels_file):
        word_index_matrix = self.load_word_index_matrix(word_index_matrix_file)
        labels = self.load_labels(labels_file)
        self.word_index_matrix_train, self.word_index_matrix_test, self.labels_train, self.labels_test = train_test_split(
            word_index_matrix,
            labels,
            test_size=0.20,
            shuffle=True,
            random_state=42)

    def _get_n_files(self):
        n_files = 0
        for dataset_dir in self.dataset_dirs:
            n_files += len(
                [name for name in os.listdir(dataset_dir) if os.path.isfile(os.path.join(dataset_dir, name))])
        return n_files

    def _load_word_list(self, word_list_file):
        if isfile(word_list_file):
            words_list = list(np.load(word_list_file))
        else:
            words_list = self.word_embedding.get_words_list()
            words_list = [word.decode('utf-8') for word in words_list]
        return words_list

    def load_word_vectors(self, embeddings_file):
        # if isfile("../word_models/word2vec_google_matrix.npy"):
        if isfile(embeddings_file):
            return self.word_embedding.load_word2vec_embeddings(embeddings_file).astype("float32")
        else:
            return self.word_embedding.create_word2vec_embeddings().astype("float32")

    def _clean_sentences(self, string):
        string = string.lower().replace("<br />", " ")
        return re.sub(strip_special_chars, "", string.lower())

    def get_train_batch(self, ):
        word_index_matrix = self.word_index_matrix_train
        batch_train = np.zeros((self.batch_size, self.max_seq_length))
        batch_train_labels = []
        trainset_size = len(self.labels_train)
        for i in range(self.batch_size):
            r = randint(0, trainset_size - 1)
            batch_train[i] = word_index_matrix[r]

            if self.labels_train[r] == 1:
                batch_train_labels.append([1, 0])
            else:
                batch_train_labels.append([0, 1])

        return batch_train, batch_train_labels

    def get_test_batch(self, ):
        word_index_matrix = self.word_index_matrix_test
        batch_test = np.zeros((self.batch_size, self.max_seq_length))
        batch_test_labels = []
        testset_size = len(self.labels_test)
        for i in range(self.batch_size):
            r = randint(0, testset_size - 1)
            batch_test[i] = word_index_matrix[r]
            if self.labels_train[r] == 1:
                batch_test_labels.append([1, 0])
            else:
                batch_test_labels.append([0, 1])

        return batch_test, batch_test_labels

    def load_word_index_matrix(self, word_index_matrix_file):
        return np.load(word_index_matrix_file)

    def load_labels(self, labels_file):
        return np.load(labels_file)

    def convert_sentence_to_word_index_vector(self, sentence):
        word_index_vector = np.zeros((self.batch_size, self.max_seq_length), dtype='int32')
        preprocessed_sentence = self._clean_sentences(sentence)
        words = preprocessed_sentence.split()
        for word_id, word in enumerate(words):
            if word_id >= self.max_seq_length:
                break

            try:
                word_index_vector[0][word_id] = self.words_list.index(word)
            except:
                word_index_vector[0][word_id] = len(self.words_list) - 1

        return word_index_vector

    def convert_batch_sentences_to_word_index_vector(self, sentences):
        word_index_vector = np.zeros((self.batch_size, self.max_seq_length), dtype='int32')
        for i, sentence in enumerate(sentences):
            preprocessed_sentence = self._clean_sentences(sentence)
            words = preprocessed_sentence.split()
            for word_id, word in enumerate(words):
                if word_id >= self.max_seq_length:
                    break

                try:
                    word_index_vector[i][word_id] = self.words_list.index(word)
                except:
                    word_index_vector[i][word_id] = len(self.words_list) - 1

        return word_index_vector

    def create_word_index_matrix(self, word_index_matrix_file, labels_file):
        word_index_matrix = np.zeros((self.n_files, self.max_seq_length), dtype='int32')
        labels = []
        file_id = 0
        non = 0
        for dirname in self.dataset_dirs:
            for file_name in os.listdir(dirname):
                for line in open(os.path.join(dirname, file_name)):
                    preprocessed_line = self._clean_sentences(line)
                    words = preprocessed_line.split()
                    for word_id, word in enumerate(words):
                        if word_id >= self.max_seq_length:
                            break
                        try:
                            word_index_matrix[file_id][word_id] = self.words_list.index(word)
                        except:
                            non += 1
                            word_index_matrix[file_id][word_id] = len(self.words_list) - 1

                label = 1 if dirname == "../dataset/positive_reviews" else -1
                labels.append(label)
                file_id += 1

        print(non)
        np.save(word_index_matrix_file, word_index_matrix)
        np.save(labels_file, labels)
        return word_index_matrix, labels

    def create_word_index_matrix_twitter(self, dataset_file, word_index_matrix_file, labels_file):
        labels = []
        line_id = 0
        non = 0
        on = 0
        sentiment_analysis_df = pd.read_csv(dataset_file, error_bad_lines=False)
        sentiment_analysis_texts = sentiment_analysis_df["SentimentText"].tolist()
        sentiment_analysis_sentiments = sentiment_analysis_df["Sentiment"].tolist()

        word_index_matrix = np.zeros((len(sentiment_analysis_sentiments), self.max_seq_length), dtype='int32')

        for sentiment, text in zip(sentiment_analysis_sentiments, sentiment_analysis_texts):

            try:
                preprocessed_line = self._clean_sentences(text)
            except:
                preprocessed_line = "test"
                sentiment = "1"
            # preprocessed_line = str(text)

            words = preprocessed_line.split()
            for word_id, word in enumerate(words):
                if word_id >= self.max_seq_length:
                    break

                try:
                    word_index_matrix[line_id][word_id] = self.words_list.index(word)
                    on += 1
                except:
                    non += 1
                    word_index_matrix[line_id][word_id] = len(self.words_list) - 1

            # print(line_id)
            line_id += 1
            label = 1 if sentiment == 1 else -1
            labels.append(label)

        print("number of non " + str(non))
        print("number of on " + str(on))
        np.save(word_index_matrix_file, word_index_matrix)
        np.save(labels_file, labels)
        return word_index_matrix, labels

    def create_word_list(self, input_file_name, output_file_name, min_count=5):
        words = []
        with open(input_file_name) as file_reader:
            for line in file_reader:
                words += line.split()

        word_counts = Counter(words)
        frequent_words = [word for word, freq in word_counts.items() if freq > min_count]
        np.save(output_file_name, frequent_words)
        return frequent_words


if __name__ == "__main__":
    # dataset = SentimentDatasetReader(max_seq_length=30)
    # dataset.create_word_index_matrix_twitter(dataset_file="../dataset/twitter/cleaned_sentiment_analysis_dataset.csv",
    #                                          word_index_matrix_file=WORD_INDEX_MATRIX_FILE,
    #                                          labels_file=LABELS_FILE)

    # wim = dataset.load_word_index_matrix("../word_models/twitter_models/twitter_word_index_matrix.npy")
    # a=1
    # dataset.create_word_list(input_file_name="../word_models/all_twitter_sentences.txt",
    #                          output_file_name="../word_models/all_twitter_words.npy",
    #                          min_count=3)

    # batch_train, batch_train_labels = dataset.get_train_batch2()
    # arr, labels = dataset.get_train_batch()
    # print(batch_train)

    # a = np.load("idsMatrix.npy")
    # b = np.load("word_models/word_index_matrix.npy")
    # vec = dataset.convert_sentence_to_word_index_vector("the movie was amazing")

    ######################################################

    # dataset = SentimentDatasetReader(dataset_dirs=["../dataset/movie_reviews/negative_reviews", "../dataset/movie_reviews/positive_reviews"],
    #                                  max_seq_length=250)
    # dataset.create_word_index_matrix(word_index_matrix_file=WORD_INDEX_MATRIX_FILE,
    #                                  labels_file=LABELS_FILE)
    #wem = np.load(WORD_EMBEDDING_FILE)

    ######################################################


    # dataset = SentimentDatasetReader(max_seq_length=30, batch_size=24)
    #
    # dataset.create_word_index_matrix_twitter(dataset_file="../dataset/tweets/cleaned_Tweets.csv",
    #                                           word_index_matrix_file=WORD_INDEX_MATRIX_FILE,
    #                                           labels_file=LABELS_FILE)
    #
    a = np.load(LABELS_FILE)
    a=1