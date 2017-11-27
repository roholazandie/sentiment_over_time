import pandas as pd
from gensim import corpora
from gensim import models
from preprocessing.lazy_corpus import FileLazyCorpus, ListLazyCorpus
from visualization.plotlyvisualize import heat_map_plot
from visualization.topic_modeling_semantic_network import visualize_semantic_netwrok
import logging
import numpy as np
from numpy import unravel_index

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class LatentSemanticIndexingModelSelection():
    def __init__(self, file_name, sentence_list=[]):
        self.min_num_topics = 5
        self.max_num_topics = 20
        if file_name:
            lazy_corpus = FileLazyCorpus(file_name)
        else:
            lazy_corpus = ListLazyCorpus(sentence_list)

        self.corpus = []
        for vector in lazy_corpus:
            self.corpus.append(vector)

        tfidf = models.TfidfModel(self.corpus)
        self.corpus_tfidf = tfidf[self.corpus]

        lazy_corpus.save_dictionary("../word_models/dictionary.dict")
        self.dictionary = corpora.Dictionary.load("../word_models/dictionary.dict")



    def find_perplexities(self):
        #perplexities = np.zeros(self.num_topics, self.num_words)
        perplexities = []
        for i, num_topic in enumerate(range(self.min_num_topics, self.max_num_topics)):
            #for j, num_word in enumerate(range(self.num_words)):
            lsi = models.LsiModel(self.corpus_tfidf, id2word=self.dictionary, num_topics=num_topic,
                                  power_iters=5)
            #perplexities[i, j] = lda.log_perplexity(self.corpus)

            perplexities.append(lsi.log_perplexity(self.corpus))


        return perplexities

if __name__ == "__main__":
    tweets_df = pd.read_csv("../word_models/tweeter_donald trump.csv")
    tweets_text = tweets_df["text"].tolist()
    with open("../word_models/tweets_donal_trump.txt", 'w') as file_writer:
        for tweet in tweets_text:
            file_writer.write(str(tweet)+"\n")

    topic_modeling_model_selection = LatentSemanticIndexingModelSelection(file_name="../word_models/tweets_donal_trump.txt")
    perplexities = topic_modeling_model_selection.find_perplexities()
    print(perplexities)
    print(perplexities.index(max(perplexities)))
    # index_max = unravel_index(perplexities.argmax(), perplexities.shape)
    # print(index_max)
    # heat_map_plot(z=perplexities, filename="outputs/perplexities.html")