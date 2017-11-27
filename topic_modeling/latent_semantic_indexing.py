from gensim import corpora
from gensim import models

from preprocessing.lazy_corpus import FileLazyCorpus, ListLazyCorpus
from topic_modeling import TopicModeling
from visualization.topic_modeling_semantic_network import visualize_semantic_netwrok


class LatentSemanticIndexing(TopicModeling):
    def __init__(self, file_name, sentence_list=[]):
        super().__init__(file_name)
        self.num_topics = 10
        self.num_topics_to_show = 5
        self.num_words = 15
        self.alpha_value = 0.1  # Smaller alpha values should give much more distinguishing topics

        if file_name:
            lazy_corpus = FileLazyCorpus(file_name)
        else:
            lazy_corpus = ListLazyCorpus(sentence_list)

        corpus = []
        for vector in lazy_corpus:
            corpus.append(vector)

        tfidf = models.TfidfModel(corpus)
        self.corpus_tfidf = tfidf[corpus]

        lazy_corpus.save_dictionary("word_models/dictionary.dict")
        self.dictionary = corpora.Dictionary.load("word_models/dictionary.dict")



    def visualize_topics(self, ):
        topics = self.get_topics()
        visualize_semantic_netwrok(topics,
                                   visualize_method='plotly',
                                   filename="outputs/lsi_out.html",
                                   title='Latent Semantic Indexing')


    def get_topics(self):
        lsi = models.LsiModel(self.corpus_tfidf, id2word=self.dictionary, num_topics=self.num_topics,
                              power_iters=5)  # initialize an LSI transformation

        topics = []

        for topic in lsi.print_topics(num_topics=self.num_topics_to_show, num_words=self.num_words):
            topics.append(
                [((item.split('*')[1]).strip(' "'), float(item.split('*')[0])) for item in topic[1].split('+')])
            print(topic)

        return topics


if __name__ == "__main__":
    # tweets_df = pd.read_csv("word_models/tweeter_donald trump.csv")
    # tweets_text = tweets_df["text"].tolist()
    # with open("word_models/tweets_donal_trump.txt", 'w') as file_writer:
    #     for tweet in tweets_text:
    #         file_writer.write(str(tweet) + "\n")

    topic_modeling = LatentSemanticIndexing(file_name="word_models/tweets_donal_trump.txt")
    topic_modeling.visualize_topics()