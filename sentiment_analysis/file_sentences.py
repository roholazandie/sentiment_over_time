import os


class FileSentences(object):
    def __init__(self, file_name):
        self.file_name = file_name



    def __iter__(self):
        for line in open(self.file_name):
            if line.strip()!="":
                yield line.split()



if __name__ == "__main__":
    # import pandas as pd
    # cleaned_sentiment_dataset_df = pd.read_csv("../word_models/cleaned_sentiment_analysis_dataset.csv", error_bad_lines=False)
    # all_twitter_sentences = cleaned_sentiment_dataset_df["SentimentText"].tolist()
    # with open("../word_models/all_twitter_sentences.txt", 'w') as file_writer:
    #     for sentence in all_twitter_sentences:
    #         file_writer.write(str(sentence)+"\n")

    file_sentences = FileSentences(file_name="../word_models/all_twitter_sentences.txt")
    for sentence in file_sentences:
        print(sentence)