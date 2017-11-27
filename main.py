'''
This is the main to the whole projec
in this main we try to run the experiments on the files and models that has been collected and
trained to make the whole process faster
if you want to change the models and try them online you should run their scripts
'''
from sentiment_analysis.twitter_sentiments import TwitterSentiments


twitter_sentiments = TwitterSentiments()
file = "word_models/tweeter_donald trump.csv"
twitter_sentiments.topic_modeling_based_on_sentiment(file_name=file ,algorithm='lsi')