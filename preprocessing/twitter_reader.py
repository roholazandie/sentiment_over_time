import pandas as pd
import tweepy

from preprocessing.twitter_cleaner import TwitterCleaner


class TwitterReader(object):

    def __init__(self):
        self.consumer_key = "vovEnWK2r87w4syfQfGJOWU4h"
        self.consumer_secret = "Q9ZDymMBRigSXVuFgKRANNsHxtAkB1pwOgV92yXQQguEuJ6495"
        self.access_token = "251150472-Xa3DMfhdq7LKKWfoXtyIEU5OBRHLQcWG8kwzJQBd"
        self.access_token_secret = "jOHhGeuyHR3XFfIxUGi5XGx3btzVmA147tiESl2DeZO4f"
        self._authenticate()
        self.api = tweepy.API(self.auth, wait_on_rate_limit=True, retry_count=2)
        self.twitter_cleaner = TwitterCleaner()


    def _authenticate(self):
        self.auth = tweepy.OAuthHandler(self.consumer_key, self.consumer_secret)
        self.auth.set_access_token(self.access_token, self.access_token_secret)


    def retrieve_hash_tag(self, hashtag, since_date, count):
        tweet_texts = []
        tweet_times = []
        tweet_locations = []
        i = 0
        for tweet in tweepy.Cursor(self.api.search,
                                   q=hashtag,
                                   lang="en",
                                   since=since_date
                                   ).items(count):
            i+=1
            if not tweet.retweeted:
                tweet_text_processed = self._preprocess(tweet.text)
                #print(tweet_text_processed)
                if tweet_text_processed in tweet_texts:
                    continue
                tweet_texts.append(tweet_text_processed)
                tweet_times.append(tweet.created_at)
                tweet_locations.append(tweet.user.location)
                # if i>10:
                #     since_date = since_date - timedelta(days=1)
                #     i=0
            else:
                print("retweeted")

        print(i)
        tweets_df = pd.DataFrame({"time": tweet_times, "text": tweet_texts, "location": tweet_locations} )
        return tweets_df


    def retrieve_trend_hashtags(self, n_top=10):
        trends1 = self.api.trends_place(1)
        data = trends1[0]
        trends = data["trends"]
        names = [trend['name'] for trend in trends]
        trendsName = ' '.join(names)
        print(trendsName)



    def _preprocess(self, text):
        text = self.twitter_cleaner.clean(text)
        return text


if __name__ == "__main__":
    twitter_reader = TwitterReader()
    tweets = twitter_reader.retrieve_hash_tag(hashtag="#iran", since_date="2017-04-03", count=100)
    a=1
    #twitter_reader.retrieve_trend_hashtags()
    #a = twitter_reader._preprocess("@iteration")
    #print(a)