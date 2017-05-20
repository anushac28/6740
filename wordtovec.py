import gensim,logging
from gensim.models.keyedvectors import KeyedVectors
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

# w = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
# #w = KeyedVectors.load_word2vec_format('1.xml', binary=False)
# w.save("vectors.txt")
# print w["queen"]
# print w["UNK"]
# print len(w["UNK"])		#300
# print w.most_similar(positive=['woman', 'king'], negative=['man'])



#https://rare-technologies.com/word2vec-tutorial/
#https://radimrehurek.com/gensim/models/word2vec.html



model = Word2Vec(LineSentence("2.xml"), size=100, window=5, min_count=1)
#workers=multiprocessing.cpu_count())
model.save('result.w2v.model')
#model.save_word2vec_format('result.w2v.vector', binary=False)
print model["Cyprus"]
print len(model["Cyprus"])
