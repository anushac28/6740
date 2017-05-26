
import pickle

class VocabMapping(object):
    def __init__(self):
        with open("/home/anusha/Desktop/CURRENT/vocab.txt", "rb") as handle:
            self.dic = pickle.loads(handle.read())

    def getIndex(self, token):
        try:
            return self.dic[token]
        except:
            return self.dic["<UNK>"]

    def getSize(self):
        return len(self.dic)
