import datetime
import time
# from extract_sentiment2 import ExtractSentiment
from itertools import zip_longest

import numpy as np
import pandas as pd
from topic_modeling.latent_dirichlet_allocation import LatentDirichletAllocation

from sentiment_analysis.extract_sentiment import ExtractSentiment
from preprocessing.twitter_reader import TwitterReader
from topic_modeling.latent_semantic_indexing import LatentSemanticIndexing


class TwitterSentiments(object):

    def __init__(self):
        self.batch_size = 24
        max_seq_length = 50#250
        n_iterations = 30000
        n_dimension = 300
        n_lstm_units = 50#64

        self.twitter_reader = TwitterReader()
        self.extract_sentiment = ExtractSentiment(batch_size=self.batch_size,
                                                  n_classes=2,
                                                  max_seq_length=max_seq_length,
                                                  n_dimensions=300,
                                                  n_lstm_units=n_lstm_units,
                                                  n_iterations=n_iterations)

        self.tweets_df = pd.DataFrame()


    def dump_tweets(self, hashtag, since_date, count=1000):

        tweets_df = self.twitter_reader.retrieve_hash_tag(hashtag=hashtag,
                                                          since_date= since_date,
                                                          count=count)
        # TODO twitter parser is slow, we need to dump file and then use it, how to make online version faster?
        tweets_df.to_csv("../word_models/tweeter_"+hashtag+".csv")



    def tag_sentences(self, file_name):
        tweets_df = pd.read_csv(file_name)
        # TODO why this produce one empty chunk at the end??
        tweet_chunks = self._chunks(tweets_df["text"], self.batch_size, padvalue="")

        all_sentiments = []
        for tweet_chunk in tweet_chunks:
            if len(tweet_chunk) == self.batch_size:
                tweet_sentiment_chunks = self.extract_sentiment.classify_batch_sentences_sentiments(tweet_chunk)
                all_sentiments += tweet_sentiment_chunks


        all_sentiments = all_sentiments[:len(tweets_df.index)]
        #all_sentiments = np.random.choice(["positive", "negative"], len(tweets_df.index))
        tweets_df["sentiment"] = all_sentiments

        positive_tweets_df = tweets_df[tweets_df["sentiment"]=="positive"]
        negative_tweets_df = tweets_df[tweets_df["sentiment"]=="negative"]

        return positive_tweets_df, negative_tweets_df


    def topic_modeling_based_on_sentiment(self, file_name, algorithm='lsi'):
        positive_tweets_df, negative_tweets_df = self.tag_sentences(file_name)
        if algorithm == "lsi":
            topic_modeling = LatentSemanticIndexing(sentence_list=positive_tweets_df["text"].tolist())
            topic_modeling.visualize_topics()
            topic_modeling = LatentSemanticIndexing(sentence_list=negative_tweets_df["text"].tolist())
            topic_modeling.visualize_topics()
        else:
            topic_modeling = LatentDirichletAllocation(sentence_list=positive_tweets_df["text"].tolist())
            topic_modeling.visualize_topics()
            topic_modeling = LatentDirichletAllocation(sentence_list=negative_tweets_df["text"].tolist())
            topic_modeling.visualize_topics()



    def sentiments_over_time(self, file_name):
        tweets_df = pd.read_csv(file_name)
        all_sentiments = np.random.choice(["positive", "negative"], len(tweets_df.index))
        tweets_df["sentiment"] = all_sentiments

        sorted_tweets_df = tweets_df.sort_values("time")
        sentiment_count_per_day_df = sorted_tweets_df.groupby([sorted_tweets_df["time"].dt.date])["sentiment"].value_counts()
        sentiment_count_per_day_df = sentiment_count_per_day_df.unstack(level='sentiment')
        print(sentiment_count_per_day_df)
        # TODO run prediction for sentiments in future



    def _chunks(self, iterable, n, padvalue=None):
        "grouper(3, 'abcdefg', 'x') --> ('a','b','c'), ('d','e','f'), ('g','x','x')"
        return map(list, zip_longest(*[iter(iterable)] * n, fillvalue=padvalue))



if __name__ == "__main__":
    # t1 = time.time()
    twitter_sentiment = TwitterSentiments()
    # since_date = datetime.date(2015, 4, 3)
    # twitter_sentiment.dump_tweets(hashtag="obama", since_date=since_date, count=10000)
    # t2 = time.time()
    # print(t2-t1)
    twitter_sentiment.tag_sentences("../word_models/tweeter_donald trump.csv")