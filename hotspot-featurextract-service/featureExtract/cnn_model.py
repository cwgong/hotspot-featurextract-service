# -*- coding: utf-8 -*-

# super class
from basic_model import BaseModel 

import tensorflow as tf

# pad_sequences(), minibatches(), service_predict_minibatches()
import data_helper 

# utils
from general_utils import Progbar

import numpy as np
import gensim
from data_helper import UNK

class CNN_Model(BaseModel):
    """Specialized class of Model for relation classify"""

    def __init__(self, config):
        super(CNN_Model, self).__init__(config)
        
        if self.config.vocab_labels_path is not None:
            # 通过 tag ---> idx 字典转化为 idx ---> tag 字典
            self.idx_to_tag = {idx: tag for tag, idx in
                               self.config.vocab_labels.items()}
        
    def build(self):
        
        with tf.name_scope('place_holder'):
            
            """Define placeholders = entries to computational graph"""
            
            # shape = (batch size, max length of sentence in all batch)
            self.word_ids = tf.placeholder(tf.int32, shape=[None, self.config.max_sequence_length], name="word_ids")
            
            # softmax
            self.labels = tf.placeholder(tf.float32, shape=[None, self.config.n_classes], name='labels') 
    
            # hyper parameters
            self.dropout = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
            
            self.lr = tf.placeholder(dtype=tf.float32, shape=[], name="lr")
            
        with tf.name_scope("embeddings"):
            
            """Defines self.word_embeddings

            If self.config.embeddings is not None and is a np array initialized
            with pre-trained word vectors, the word embeddings is just a look-up
            and we don't train the vectors. Otherwise, a random matrix with
            the correct shape is initialized.
            """
            
            if self.config.embeddings is None:
                self.logger.info("INFO: randomly initializing word vectors")
                _word_embeddings = tf.Variable(
                                        tf.random_uniform([self.config.vocab_word_size - 1, self.config.embedding_size], -1.0, 1.0),
                                        name="_word_embeddings")
                _word_embeddings = tf.concat([tf.zeros([1, self.config.embedding_size]), _word_embeddings], axis=0)
                embedding_size = self.config.embedding_size
            else:
                self.logger.info("INFO: pre-trained initializing word vectors")
                model = gensim.models.Word2Vec.load(self.config.w2v_words)
                # 需要和自己构建的词典 word id 保持一致
                vocab = {}
                vocab[UNK] = 0    
                for word in model.wv.vocab.keys():
                    vocab[word] = len(vocab)
                _word_embeddings = np.zeros((self.config.vocab_word_size, self.config.dim_word_gensim), dtype='float32')
                for word, ids in vocab.items():
                    if ids == 0:
                        _word_embeddings[ids] = np.zeros(self.config.dim_word_gensim)
                    else:
                        _word_embeddings[ids] = model.wv[word]
                
                # pre_train un-trainable or trainable
                _word_embeddings = tf.Variable(
                        _word_embeddings,
                        name="_word_embeddings",
                        dtype=tf.float32,
                        trainable=self.config.embeddings_trainable)
                embedding_size = self.config.dim_word_gensim
            
            # The result of the embedding operation is a 3-dimensional tensor of 
            # shape [None, max_sequence_length, embedding_size].
            word_embeddings = tf.nn.embedding_lookup(_word_embeddings, self.word_ids, name="word_embeddings")
            
            # TensorFlow’s convolutional conv2d operation expects a 4-dimensional tensor 
            # with dimensions corresponding to batch, width, height and channel. 
            # The result of our embedding doesn’t contain the channel dimension,
            # so we add it manually, leaving us with a layer of shape
            # [None, max_sequence_length, embedding_size, 1]
            # the last dimension is channel size
            self.word_embeddings_expanded = tf.expand_dims(word_embeddings, -1)
        
        
        # Now we’re ready to build our convolutional layers followed by max-pooling. 
        # Remember that we use filters of different sizes. Because each convolution 
        # produces tensors of different shapes we need to iterate through them, 
        # create a layer for each of them, and then merge the results into one big 
        # feature vector.
        
        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(self.config.filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                """Defines Convolution Layer

                Each filter slides over the whole embedding, but varies in how many words it covers.
                """
                
                # W = [filter_size, embedding_size, channel size, dim_filters]
                # size of pool = [1, sequence_length - filter_size + 1, 1, 1]

                filter_shape = [filter_size, embedding_size, 1, self.config.dim_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.config.dim_filters]), name="b")
                conv = tf.nn.conv2d(        
                    self.word_embeddings_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # 激活函数,用的Relu,让显著地越显著
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

                # Performing max-pooling over the output of a specific filter size 
                # leaves us with a tensor of shape [batch_size, 1, 1, dim_filters].
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.config.max_sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)
        
        with tf.name_scope("conv_outputs"):
            """Defines After Convolution Layer

            Each filter slides over the whole embedding, but varies in how many words it covers.
            """
            # 获得总纬度
            num_filters_total = self.config.dim_filters * len(self.config.filter_sizes)
           
            # concat 以及 reshape 处理
            self.h_pool = tf.concat(pooled_outputs, 3)
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        
        with tf.name_scope("dropout"):
            """Defines Dropout Layer
            """
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout)
        
        with tf.name_scope("proj"):
            """Defines Projection Layer
            """
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, self.config.n_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[self.config.n_classes]), name="b")
           
            # Keeping track of l2 regularization loss (optional)
            l2_loss = tf.constant(0.0)
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            
            self.logits = tf.nn.xw_plus_b(self.h_drop, W, b, name="logits")
            
        with tf.name_scope("pred"):
            """Defines Prediction Layer
            """
            
            # 多目标分类
            #self.preds_prob = tf.nn.sigmoid(self.logits, name="preds_prob")
            # 单目标分类
            self.preds_prob = tf.nn.softmax(self.logits, name="preds_prob")
            self.preds = tf.argmax(self.preds_prob, 1, name="preds")
            
        with tf.name_scope("loss"):
            """Defines Loss
            Calculate Mean cross-entropy loss
            """
            # 多目标分类
            #self.losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.labels)
            # 单目标分类
            self.losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels)
            
            l2_reg_lambda=0.0
            self.loss = tf.reduce_mean(self.losses) + l2_reg_lambda * l2_loss

        with tf.name_scope("accuracy"):
            """Defines Accuracy
            Calculate Mean cross-entropy loss
            """
            # StackOverFlow：
            # Any tensor returned by Session.run or eval is a NumPy array.
            # 仅对单目标多分类
            correct_predictions = tf.equal(self.preds, tf.argmax(self.labels, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
        
        # ======================================================================
        # ======================================================================
        # ======================================================================
        
        # for tensorboard
        tf.summary.scalar("loss", self.loss)
        
        # Generic functions that add training op and initialize session
        self.add_train_op(self.config.lr_method, self.lr, self.loss,
                self.config.clip)
        
        # now self.sess is defined and vars are init
        self.initialize_session() 
        
        self.logger.info("INFO: cnn model build success! ")
    
    def get_feed_dict(self, words, labels=None, lr=None, dropout=None):
        """Given some data, pad it and build a feed dictionary

        Args:
            words: list of sentences. A sentence is a list of ids of a list of
                   words. A word is a list of ids
            labels: list of ids
            lr: (float) learning rate
            dropout: (float) keep prob
            注意：
                若加入词性或是字意，则需判断 words 是否还可以 *zip  
                e.g token、char、ansj_tag
        Returns:
            dict {placeholder: value}

        """
        # perform padding of the given data
        word_ids, sequence_lengths = data_helper.pad_sequences(words, 0, max_length=self.config.max_sequence_length)

        # build feed dictionary
        feed = {
            self.word_ids: word_ids
        }

        if labels is not None:
            feed[self.labels] = labels

        if lr is not None:
            feed[self.lr] = lr

        if dropout is not None:
            feed[self.dropout] = dropout
        
        # sequence_lengths 用于 lstm
        return feed, sequence_lengths
    
    def predict_batch(self, words):
        """
        Args:
            words: list of sentences

        Returns:
            labels_pred: list of labels for each sentence
            sequence_length

        """
        fd, sequence_lengths = self.get_feed_dict(words, dropout=1.0)
        
        preds, preds_prob = self.sess.run([self.preds, self.preds_prob], feed_dict=fd)

        return preds, preds_prob
    
    def predict_dev_batch(self, words, labels):
        """
        Args:
            words: list of sentences

        Returns:
            labels_pred: list of labels for each sentence
            sequence_length

        """
        fd, sequence_lengths = self.get_feed_dict(words, labels=labels, dropout=1.0)
        
        preds, preds_prob, accuracy = self.sess.run([self.preds, self.preds_prob, self.accuracy], feed_dict=fd)

        return preds, preds_prob, accuracy
    
    def run_evaluate(self, test):
        """Evaluates performance on test set

        Args:
            test: dataset that yields tuple of (sentences, labels)

        Returns:
            metrics: (dict) metrics["acc"] = 98.4, ...

        """
        # 二分类
        if self.config.n_classes == 3:
            
            accs = []
            
            qt, gs, gd, qt_gs, qt_gd, gs_qt, gs_gd, gd_qt, gd_gs = 0, 0, 0, 0, 0, 0, 0, 0, 0
            
            for words, labels in data_helper.minibatches(test, self.config.predict_batch_size):
                
                preds, preds_prob, accuracy = self.predict_dev_batch(words,labels)
                
                accs.append(accuracy)
                
                # for i in batch size
                # zip 用来分解 batch size
                for lab, lab_pred in zip(labels, preds):

                    lab = np.argmax(lab)
                    
                    if lab == 0:
                        if lab == lab_pred:
                            qt += 1
                        else:
                            if lab_pred == 1:
                                qt_gs += 1
                            else:
                                qt_gd += 1
                                
                    elif lab == 1:
                        if lab == lab_pred:
                            gs += 1
                        else:
                            if lab_pred == 0:
                                gs_qt += 1
                            else:
                                gs_gd += 1             
                                
                    elif lab == 2:
                        if lab == lab_pred:
                            gd += 1
                        else:
                            if lab_pred == 0:
                                gd_qt += 1
                            else:
                                gd_gs += 1             
            
            qt_recall = qt/(qt+qt_gs+qt_gd) if qt > 0 else 0
            gs_recall = gs/(gs+gs_qt+gs_gd) if gs > 0 else 0
            gd_recall = gd/(gd+gd_qt+gd_gs) if gd > 0 else 0
                        
            qt_acc = qt/(qt+gs_qt+gd_qt) if qt > 0 else 0
            gs_acc = gs/(gs+qt_gs+gd_gs) if gs > 0 else 0
            gd_acc = gd/(gd+qt_gd+gs_gd) if gd > 0 else 0
            
            self.logger.info('qt: ' + str(qt+qt_gs+qt_gd) + ' gs: ' + str(gs+gs_qt+gs_gd) + ' gd: ' + str(gd+gd_qt+gd_gs))
            self.logger.info('qt_gs: ' + str(qt_gs) + ' qt_gd: ' + str(qt_gd) + ' gs_qt: ' + str(gs_qt)+ ' gs_gd: ' + str(gs_gd) + ' gd_qt: ' + str(gd_qt) + ' gd_qs: ' + str(gd_gs))
            
            self.logger.info('qt_recall: ' + str(qt_recall) + ' gs_recall: ' + str(gs_recall) + ' gd_recall: ' + str(gd_recall))
            self.logger.info('qt_acc: ' + str(qt_acc) + ' gs_acc: ' + str(gs_acc) + ' gd_acc: ' + str(gd_acc))
            
            acc = np.mean(accs)
            
            index_judge = np.mean([gs_recall, gd_recall, gs_acc, gd_acc, acc])
            return {"acc": 100*acc, "index_judge": index_judge}
        
        # 多分类
        else:
            accs = []
            for words, labels in data_helper.minibatches(test, self.config.predict_batch_size):
                
                preds, preds_prob, accuracy = self.predict_dev_batch(words,labels)
                accs.append(accuracy)
                
            acc = np.mean(accs)
            
            return {"acc": 100*acc, "f1": 100*acc}
        
    # yield 每一个 epoch 都 shuffle？ 未验证
    def run_epoch(self, train, dev, epoch):
        """Performs one complete pass over the train set and evaluate on dev

        Args:
            train: dataset that yields tuple of sentences, labels
            dev: dataset
            epoch: (int) index of the current epoch

        Returns:
            f1: (python float), score to select model on, higher is better

        """
        # progbar stuff for logging
        batch_size = self.config.batch_size
        nbatches = (len(train) + batch_size - 1) // batch_size
                   
        prog = Progbar(target=nbatches)
        
        if self.config.debug:
            self.logger.info("INFO: Tensorflow Model Debug Mode run... ")
            # iterate over dataset
            for i, (words, labels) in enumerate(data_helper.minibatches(train, batch_size)):
                fd, _ = self.get_feed_dict(words, labels, self.config.lr,
                        self.config.dropout)
    
                _, loss, word_ids, labels,logits, preds_prob, preds= self.sess.run(
                        [self.train_op, self.loss, self.word_ids, self.labels, self.logits, self.preds_prob, self.preds], feed_dict=fd)
               
                print('\nword_ids: ', word_ids)
                print('\nword_ids shape: ', word_ids.shape)
                print('\nlabels: ', labels)
                print('\nlabels shape: ', labels.shape)
                print('\nlogits: ', logits)
                print('\nlogits shape: ', logits.shape)
                print('\npreds_prob: ', preds_prob)
                print('\npreds: ', preds)
                print('\ntrain_loss: ', loss)
        else:
            # iterate over dataset
            for i, (words, labels) in enumerate(data_helper.minibatches(train, batch_size, shuffle = True)):
                fd, _ = self.get_feed_dict(words, labels, self.config.lr,
                        self.config.dropout)
    
                _, train_loss, summary, acc = self.sess.run(
                        [self.train_op, self.loss, self.merged, self.accuracy], feed_dict=fd)
    
                prog.update(i + 1, [("train loss", train_loss)])
    
                self.logger.info('batch step: '+ str(i+1) + ' loss: ' + str(train_loss) + ' acc: ' + str(acc))
    
                # tensorboard
                if i % 10 == 0:
                    self.file_writer.add_summary(summary, epoch*nbatches + i)
        
        if self.config.nepoch_no_imprv is None:
            # 每次都存
            self.save_session()
            self.logger.info("INFO: Save Session Success! ")
        
        metrics = self.run_evaluate(dev)
        
        msg = " - ".join(["{} {:04.2f}".format(k, v)
                for k, v in metrics.items()])
    
        self.logger.info(msg)

        return metrics["index_judge"]
        
            
    def train(self, train, dev):
        """Performs training with early stopping and lr exponential decay

        Args:
            train: dataset that yields tuple of (sentences, labels)
            dev: dataset

        """
        best_score = 0
        # for early stoppings
        nepoch_no_imprv = 0 
        
        # tensorboard 生成文件 path ---> self.config.dir_output
        self.add_summary() 

        for epoch in range(self.config.nepochs):
            self.logger.info("Epoch {:} out of {:}".format(epoch + 1,
                        self.config.nepochs))
            
            # acc f1
            score = self.run_epoch(train, dev, epoch)
            
            # decay learning rate
            self.config.lr *= self.config.lr_decay 

            # early stopping and saving best parameters
            if score >= best_score:
                nepoch_no_imprv = 0
                self.save_session()
                self.logger.info("INFO: Save Session Success! ")
                best_score = score
                self.logger.info("- new best score!")
            else:
                nepoch_no_imprv += 1
                if nepoch_no_imprv >= self.config.nepoch_no_imprv:
                    self.logger.info("- early stopping {} epochs without "\
                            "improvement".format(nepoch_no_imprv))
                    self.logger.info("- best score! --> " + str(best_score))
                    break

    
    def predicts(self, data):
        """Returns list of pred
        Special Sign Version:

        Args:
            words_raw: dic = {'segment_list':  ,
                              'industry_start':  ,
                              'industry_end':  ,
                              'stock_start':  ,
                              'stock_end':  }
        Returns:
            preds_prob
        
        targetInsutry targetStock Version:

        Args:
            words_raw: dic = {'segment_list':  ,
                              'industryStocks': [[industryStart,industryEnd,stockStart,stockEnd],[],[],[] ...]
                              }
        Returns:
            preds, preds_prob
            preds_prob = [0.33..., 0.68...]
            
        """
        batches = data_helper.service_predict_minibatches(data, self.config.predict_batch_size)

            
        preds, preds_prob = self.predict_batch(batches)

        return preds   

    # 没有 label 的 test
    def service_predicts(self, test):
        """service_predicts:
            
            e.g.
            
            Special Sign Version:
            # test:
                data set, type = iterator
                one_raw = {'segment_list':  ,
                           'industry_start':  ,
                           'industry_end':  ,
                           'stock_start':  ,
                           'stock_end':  }
                
            # 1. batch in batches
                words in service_predict_minibatches(test, predict_batch_size) 
                
            # 2. batch process...
                predict_batch(words)
                    1. get_feed_dict()
                        pad_sequences()
                            word_ids, sequence_lengths
                            
                    2. labels_pred = sess.run(self.marginals_op, feed_dict=fd)
                    
            # 3. get all batch predictions
            
        """
        # service predicts logger
        self.logger.info('Tensorflow Session ...')
        
        labels_preds = []
        
        count = 0
        
        batches = data_helper.service_predict_minibatches(test, self.config.predict_batch_size)
    
        for words in batches:
            count += 1
            self.logger.info(str(count) + ' sess run ...')
            labels_pred,_ = self.predict_batch(words)
            self.logger.info(labels_pred)
            self.logger.info(str(count) + ' sess finished!')
            
            
            for label in labels_pred:    
                labels_preds.append(label)

        return labels_preds
        