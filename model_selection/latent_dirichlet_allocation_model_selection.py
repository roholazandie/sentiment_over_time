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

class LatentDirichletAllocationModelSelection():
    def __init__(self, file_name="", sentence_list=[]):
        self.min_num_topics = 5
        self.max_num_topics = 30
        self.num_words = 15
        self.alpha_value = 0.01  # Smaller alpha values should give much more distinguishing topics
        if file_name:
            lazy_corpus = FileLazyCorpus(file_name)
        else:
            lazy_corpus = ListLazyCorpus(sentence_list)

        self.corpus = []
        for vector in lazy_corpus:
            self.corpus.append(vector)

        lazy_corpus.save_dictionary("../word_models/dictionary.dict")
        self.dictionary = corpora.Dictionary.load("../word_models/dictionary.dict")



    def find_perplexities(self):
        #perplexities = np.zeros(self.num_topics, self.num_words)
        perplexities = []
        for i, num_topic in enumerate(range(5, self.max_num_topics)):
            #for j, num_word in enumerate(range(self.num_words)):
            lda = models.ldamodel.LdaModel(corpus=self.corpus, id2word=self.dictionary, num_topics=num_topic,
                                       update_every=1, passes=6, alpha='auto', eta='auto')
            #perplexities[i, j] = lda.log_perplexity(self.corpus)
            perplexities.append(lda.log_perplexity(self.corpus))


        return perplexities

if __name__ == "__main__":
    sentiment_analysis_df = pd.read_csv("../word_models/Sentiment Analysis Dataset.csv", error_bad_lines=False)
    sentiment_analysis_texts = sentiment_analysis_df["SentimentText"].tolist()

    topic_modeling_model_selection = LatentDirichletAllocationModelSelection(sentence_list=sentiment_analysis_texts)
    perplexities = topic_modeling_model_selection.find_perplexities()
    print(perplexities)
    print(perplexities.index(min(perplexities)))
    # index_max = unravel_index(perplexities.argmax(), perplexities.shape)
    # print(index_max)
    # heat_map_plot(z=perplexities, filename="outputs/perplexities.html")