import re as regex

class TwitterCleaner(object):
    def __init__(self):
        pass

    def iterate(self):
        for cleanup_method in [self.remove_urls,
                               self.remove_usernames,
                               self.remove_special_chars,
                               self.remove_numbers]:
            yield cleanup_method


    def clean(self, text):
        for method in self.iterate():
            text = method(text)
        return text


    @staticmethod
    def remove_by_regex(tweets, regexp):
        tweets = regex.sub(regexp, "", tweets)
        return tweets


    def remove_urls(self, tweets):
        return self.remove_by_regex(tweets, regex.compile(r"http.?://[^\s]+[\s]?"))


    def remove_special_chars(self, tweets):  # it unrolls the hashtags to normal words
        for remove in map(lambda r: regex.compile(regex.escape(r)), [",", ":", "\"", "=", "&", ";", "%", "$",
                                                                     "@", "%", "^", "*", "(", ")", "{", "}",
                                                                     "[", "]", "|", "/", "\\", ">", "<", "-",
                                                                     "!", "?", ".", "'",
                                                                     "--", "---", "#"]):
            tweets = regex.sub(remove, "", tweets)
        return tweets


    def remove_usernames(self, tweets):
        return self.remove_by_regex(tweets, regex.compile(r"@[^\s]+[\s]?"))


    def remove_numbers(self, tweets):
        return self.remove_by_regex(tweets, regex.compile(r"\s?[0-9]+\.?[0-9]*"))