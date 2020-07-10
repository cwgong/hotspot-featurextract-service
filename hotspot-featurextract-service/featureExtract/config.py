# -*- coding: utf-8 -*-

import os
import data_helper
from general_utils import get_logger

class Config():
    
    def __init__(self, load=True):
        """Initialize hyperparameters and load vocabs

        Args:
            load_embeddings: (bool) if True, load embeddings into
                np array, else None

        """

        # load if requested (default)
        if load:
            
            # directory for training outputs
            if not os.path.exists(self.dir_output):
                os.makedirs(self.dir_output)
            
    
            # create instance of logger
            self.logger = get_logger(self.path_log)
        
            self.load()


    def load(self):
        """Loads vocabulary, processing functions and embeddings

        Supposes that build_data.py has been run successfully and that
        the corresponding files have been created (vocab and trimmed GloVe
        vectors)

        """
        # 1. vocabulary
        self.vocab_words = data_helper.load_vocab(self.vocab_words_path)
        self.vocab_word_size     = len(self.vocab_words)
        
        if self.vocab_labels_path is not None:
            self.vocab_labels  = data_helper.load_vocab(self.vocab_labels_path)
            self.n_classes  = len(self.vocab_labels)
            # 2. get processing functions that map label -> id
            self.processing_y  = data_helper.get_processing(self.vocab_labels,lowercase=False, allow_unk=False)
        else:
            self.n_classes  = 3

        # 2. get processing functions that map word -> id
        self.processing_x = data_helper.get_processing(self.vocab_words, lowercase=True, allow_unk=True)


    dir_output = "model_titlencoding_artifical/"
    dir_model  = dir_output + "cnn_model/"
    path_log   = dir_output + "log.txt"
    
    # dataset config
    max_iter = None # None
    filename_train = './data/train_.json'
    filename_dev = './data/dev_.json'
    
    # vocab config
    embeddings = None # 是否为预训练词向量 # 构建词典时也需要用到
    vocab_words_path = './data/vocab/words_vocab.txt'
    vocab_labels_path = None
    
    # embeddings
    if embeddings == None:
        embedding_size = 128
    
    # 在 gensim 中会调用此 参数
    else:
        w2v_words = './gensim/w2v/words_64'
        dim_word_gensim = 64
        embeddings_trainable = True
        
    # cnn params
    max_sequence_length = 303
    filter_sizes = [2,3,4]
    dim_filters = 256
    
        
    # training
    nepochs          = 50
    dropout          = 0.5
    batch_size       = 16
    predict_batch_size       = 256
    lr_method        = "adam"
    lr               = 0.001
    lr_decay         = 0.9
    clip             = -1 # if negative, no clipping
    
    # 如果超过 nepoch_no_imprv 个 epoch 没有提高，则提前停止
    nepoch_no_imprv  = 50
    
    debug = False
