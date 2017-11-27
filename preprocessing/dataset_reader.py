import os
import re
from os.path import isfile
from random import randint

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sentiment_analysis.word_embedding import WordEmbedding

strip_special_chars = re.compile("[^A-Za-z0-9 ]+")


class SentimentDatasetReader(object):

    def __init__(self, batch_size, max_seq_length):
        self.dataset_dirs = ["dataset/negative_reviews", "dataset/positive_reviews"]
        self.word_embedding = WordEmbedding(directories=self.dataset_dirs,
                                       model_file="word_models/word2vec.bin",
                                       dimension=100)
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.n_files = self._get_n_files()
        self.word_vectors = self.load_word_vectors()
        self.words_list = self._load_word_list()
        if isfile("word_models/word_index_matrix_google.npy"):
            self.dataset_train_test_split()


    def dataset_train_test_split(self):
        word_index_matrix = self.load_word_index_matrix()
        labels = self.load_labels()
        self.word_index_matrix_train, self.word_index_matrix_test, self.labels_train, self.labels_test = train_test_split(word_index_matrix,
                                                            labels,
                                                            test_size = 0.10,
                                                            shuffle=True,
                                                            random_state = 42)


    def _get_n_files(self):
        n_files = 0
        for dataset_dir in self.dataset_dirs:
            n_files += len([name for name in os.listdir(dataset_dir) if os.path.isfile(os.path.join(dataset_dir, name))])
        return n_files


    def _load_word_list(self):
        if isfile("word_models/all_words.npy"):
            words_list = list(np.load("word_models/all_words.npy"))
        else:
            words_list = self.word_embedding.get_words_list()
            words_list = [word.decode('utf-8') for word in words_list]
        return words_list


    def load_word_vectors(self):
        if isfile("word_models/word2vec_google_matrix.npy"):
            return self.word_embedding.load_word2vec_embeddings().astype("float32")
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
            r = randint(0, trainset_size-1)
            batch_train[i] = word_index_matrix[r]
            if self.labels_train[r] == 1:
                batch_train_labels.append([1, 0])
            else:
                batch_train_labels.append([0, 1])

        return batch_train, batch_train_labels



    def get_test_batch(self, ):
        return


    def load_word_index_matrix(self):
        return np.load("word_models/word_index_matrix_google.npy")


    def load_labels(self):
        return np.load("word_models/labels_google.npy")


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
                word_index_vector[0][word_id] = len(self.words_list)-1

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


    def create_word_index_matrix(self):
        word_index_matrix = np.zeros((self.n_files, self.max_seq_length), dtype='int32')
        labels = []
        file_id = 0
        non=0
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
                            non+=1
                            word_index_matrix[file_id][word_id] = len(self.words_list)-1

                label = 1 if dirname=="dataset/positive_reviews" else -1
                labels.append(label)
                file_id += 1

        print(non)
        np.save("word_models/word_index_matrix_google.npy", word_index_matrix)
        np.save("word_models/labels_google.npy", labels)
        return word_index_matrix, labels


    def create_word_index_matrix_twitter(self):
        labels = []
        line_id = 0
        non = 0
        sentiment_analysis_df = pd.read_csv("word_models/Sentiment Analysis Dataset.csv", error_bad_lines=False)
        sentiment_analysis_texts = sentiment_analysis_df["SentimentText"].tolist()
        sentiment_analysis_sentiments = sentiment_analysis_df["Sentiment"].tolist()

        word_index_matrix = np.zeros((len(sentiment_analysis_sentiments), self.max_seq_length), dtype='int32')

        for sent, text in zip(sentiment_analysis_sentiments, sentiment_analysis_texts):
            preprocessed_line = self._clean_sentences(text)
            words = preprocessed_line.split()
            for word_id, word in enumerate(words):
                if word_id >= self.max_seq_length:
                    break

                try:
                    word_index_matrix[line_id][word_id] = self.words_list.index(word)
                except:
                    non += 1
                    word_index_matrix[line_id][word_id] = len(self.words_list) - 1

            #print(line_id)
            line_id +=1
            label = 1 if sent == 1 else -1
            labels.append(label)

        print("number of non"+str(non))
        np.save("word_models/word_index_matrix_twitter.npy", word_index_matrix)
        np.save("word_models/labels_twitter.npy", labels)
        return word_index_matrix, labels



if __name__ == "__main__":
    dataset = SentimentDatasetReader(batch_size=24, max_seq_length=250)
    s = dataset.create_word_index_matrix_twitter()

    #batch_train, batch_train_labels = dataset.get_train_batch2()
    # arr, labels = dataset.get_train_batch()
    #print(batch_train)

    # a = np.load("idsMatrix.npy")
    # b = np.load("word_models/word_index_matrix.npy")
    #vec = dataset.convert_sentence_to_word_index_vector("the movie was amazing")
