import pandas as pd
from gensim import corpora
from gensim import models

from preprocessing.lazy_corpus import FileLazyCorpus, ListLazyCorpus
from topic_modeling import TopicModeling
from visualization.topic_modeling_semantic_network import visualize_semantic_netwrok
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class LatentDirichletAllocation(TopicModeling):
    def __init__(self, file_name="", sentence_list=[]):
        super().__init__(file_name)
        self.scaling_factor = 10 # scaling factor is for making the lda numbers bigger for visualization
        self.num_topics = 10
        self.num_topics_to_show = 5
        self.num_words = 15
        self.alpha_value = 0.01  # Smaller alpha values should give much more distinguishing topics
        if file_name:
            lazy_corpus = FileLazyCorpus(file_name)
        else:
            lazy_corpus = ListLazyCorpus(sentence_list)

        self.corpus = []
        for vector in lazy_corpus:
            self.corpus.append(vector)

        lazy_corpus.save_dictionary("word_models/dictionary.dict")
        self.dictionary = corpora.Dictionary.load("word_models/dictionary.dict")



    def visualize_topics(self, ):
        topics = self.get_topics()
        visualize_semantic_netwrok(topics, visualize_method='plotly',
                                       filename="outputs/lda_out.html",
                                       title="Latent Dirichlet Analysis")


    def get_topics(self):
        lda = models.ldamodel.LdaModel(corpus=self.corpus, id2word=self.dictionary, num_topics=self.num_topics,
                                       update_every=1, passes=6, alpha='auto', eta='auto')
        topics = []

        for topic in lda.print_topics(num_topics=self.num_topics_to_show, num_words=self.num_words):
            topics.append([(item.split('*')[1].replace("\"", "").strip(), float(item.split('*')[0])*self.scaling_factor) for item in
                           topic[1].split('+')])
            #print(topic)

        return topics

if __name__ == "__main__":
    tweets_df = pd.read_csv("word_models/tweeter_donald trump.csv")
    tweets_text = tweets_df["text"].tolist()
    with open("word_models/tweets_donal_trump.txt", 'w') as file_writer:
        for tweet in tweets_text:
            file_writer.write(str(tweet)+"\n")

    topic_modeling = LatentDirichletAllocation(file_name="word_models/tweets_donal_trump.txt")
    topic_modeling.visualize_topics()