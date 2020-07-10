# -*- coding: utf-8 -*-

import sys 
sys.path.append('../')

from config import Config
from data_helper import Dataset_SpecialSign_version, Dataset_TargetOther_version

# 导入方法
import gensim_utils

# 验证路径
import os.path
#os.path.isdir()
#print(os.path.exists('.' + config.filename_train))
#print(os.path.exists('.' + config.w2v_words))

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

if __name__ == '__main__':
    
    # 用同样的配置
    # 注意 config 中 embeddings is not None
    config = Config(load=False)
    filename_train = '.' + config.filename_train
    model_path = '.' + config.w2v_words
    
    train = Dataset_SpecialSign_version(filename_train, max_iter = config.max_iter, gensim = True)
    
    # 训练
    print('training start ...')
    gensim_utils.word2vec_train(train, config.dim_word_gensim, model_path)
    print('train finished!')
    
    # 打印词典大小
    gensim_utils.word2vec_info(model_path)
    
