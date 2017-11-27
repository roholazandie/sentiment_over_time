import os


class Sentences(object):
    def __init__(self, dirnames):
        self.dirnames = dirnames



    def __iter__(self):
        for dirname in self.dirnames:
            for fname in os.listdir(dirname):
                for line in open(os.path.join(dirname, fname)):
                    if line.strip()!="":
                        yield line.split()



if __name__ == "__main__":
    sentces = Sentences(["negativeReviews", "positiveReviews"])
    i = 0
    for s in sentces:
        print(s)
        i+=1
        if i>10:
            break