# -*- coding: utf-8 -*-

import gensim
import numpy as np
import tensorflow as tf

model = gensim.models.Word2Vec.load('./temp/gensim_model')

# 需要将 Gensim 中的 vocab 导出作为 tensorflow model 的 vocab
# 构建 Gensim vocab

vocab = {}
vocab['UNK'] = 0
     
for word in model.wv.vocab.keys():
    vocab[word] = len(vocab)

gensim_vocab_size = len(model.wv.vocab.keys())


dim_embedding = 64
word_embeddings = np.zeros((gensim_vocab_size + 1, dim_embedding), dtype='float32')
print(type(word_embeddings))

for word, ids in vocab.items():
    if ids == 0:
        word_embeddings[ids] = np.zeros(dim_embedding)
    else:
        word_embeddings[ids] = model.wv[word]

# pre_train or trainable
word_embeddings = tf.Variable(
        word_embeddings,
        name="word_embeddings",
        dtype=tf.float32,
        trainable=True)

x_input = tf.placeholder(tf.int32, shape=[None, None])

x_embedding = tf.nn.embedding_lookup(word_embeddings, x_input)


with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        feed_dict = {x_input: [[0,1,2],[3,0,1],[1,2,3166]]}
        x_embedding_= sess.run(
                    [x_embedding],
                    feed_dict)
        print(x_embedding_)

print(vocab['白酒'])
print(model.wv['白酒'])

x = model.wv['白酒']
print(np.array(x))



