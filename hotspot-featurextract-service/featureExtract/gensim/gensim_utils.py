# -*- coding: utf-8 -*-

import gensim
import numpy as np

# param:
#    size=100, window=5, min_count=5, workers=4, sg=0
def word2vec_train(sentences, dim, model_path, workers=12):
    
    model = gensim.models.Word2Vec(sentences, size=dim, min_count=0, sg=0, workers = workers)
    model.save(model_path)
    
    print("word2vec trained successful!")
    
def word2vec_info(model_path):
    
    model = gensim.models.Word2Vec.load(model_path)
    vocab_size = len(model.wv.vocab.keys())
    print("word2vec vocab size: ",vocab_size)
    
def get_word_vector(model_path, word):
    
    model = gensim.models.Word2Vec.load(model_path)
    vector = model.wv[word]
    print("vector: ",vector)

def get_word_most_similar(model_path, word, top_n):
    
    model = gensim.models.Word2Vec.load(model_path)
    words = model.wv.most_similar(word, topn=top_n)
    print("words: ",words)

def get_word_similarity(model_path, word1, word2):
    
    model = gensim.models.Word2Vec.load(model_path)
    similarity = model.similarity(word1, word2)
    print('similarity: ',similarity)
    
