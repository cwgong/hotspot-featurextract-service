# -*- coding: utf-8 -*-

import json
import numpy as np
import requests
import logging

# shared global variables to be imported from model also
UNK = "$UNK$"
NUM = "$NUM$"

# hot spot description and influence/comment/point of view
# special_signs = ['~~[[1', '1]]~~', '~~[[2', '2]]~~']
class Dataset_HotSpot_title_encoding_artificial(object):
    """Class that iterates over Dataset

    __iter__ method yields a tuple (words, label)
        words: list of raw words
        label: label

    If processing_word and processing_tag are not None,
    optional preprocessing is appplied

    Example:
        ```python
        data = Dataset(filename)
        for sentence, labels in data:
            pass
        ```

    """
    def __init__(self, filename, processing_x=None, processing_y=None,
                 max_iter=None, gensim = False):
        """
        Args:
            filename: path to the file
            processing_x: (optional) function that takes a word as input
            processing_y: (optional) function that takes a label as input
            max_iter: (optional) max number of sentences to yield
        """
        self.filename = filename
        self.processing_x = processing_x
        self.processing_y = processing_y
        self.max_iter = max_iter
        self.length = None
        
        self.gensim = gensim
        
    # iterator
    def __iter__(self):
        '''
        e.g.
        yield 每条数据----> 
        (['word1', 'word2', 'word3', 'word4', 'word5', 'word6'], 1.0)
        (['word1', 'word2', 'word3', 'word4', 'word5', 'word6'], [0, 1])
        (['word1', 'word2', 'word3', 'word4', 'word5', 'word6'], 8)
        '''
        # 当前在 iterator 中的 sentences 个数
        niter = 0
        
        with open(self.filename, 'r', encoding = 'utf-8') as f1:
            data = json.load(f1) 
            
            for num, dic in data.items():
                
                title_segments = dic['title_segments']
                content_segments = dic['content_segments']
                
                #title_segments = self.hanlp_rough_segment(title)
                #content_segments = self.hanlp_rough_segment(content)
                
                x  = self.title_encoding_artificial(title_segments, content_segments)
                
                tag = dic['label']
                if tag == "其它":
                    tag = [1.0, 0.0, 0.0]
                elif tag == "概述":
                    tag = [0.0, 1.0, 0.0]
                elif tag == "观点":
                    tag = [0.0, 0.0, 1.0]
                    
                # 一条数据
                words = []
                for i in range(len(x)):
                    
                    # x
                    word = x[i]
                    # strio() 为构建字典不重复，与 load_vocab 相呼应
                    word = word.strip()
                    
                    if self.processing_x is not None:
                        word = self.processing_x(word)
                        
                    words += [word]

                if self.processing_y is not None:
                    tag = self.processing_y(tag)
                    
                if self.gensim == False:
                    yield words, tag
                else:
                    yield words
                    
                niter += 1
                if self.max_iter is not None and niter > self.max_iter:
                    break

    # 重写了 len(iterator)
    def __len__(self):
        """Iterates once over the corpus to set and store length"""
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1

        return self.length
     
    # 分词后 的 title_segment_list 去掉了 “ ”
    def title_segment_list_filter(self, title_segment_list):
        
        title_segment_list_ = []
        stop_words = ['的', '：','！']
        for title_segment in title_segment_list:
            if title_segment not in stop_words and title_segment.strip() != "":
                title_segment_list_.append(title_segment)
        
        return title_segment_list_
    
    # 分词后 的 content_segment_list 去掉了 “ ”
    def title_encoding_artificial(self, title_segment_list, content_segment_list):
        
        keywords_dic = {}
        title_segment_list = self.title_segment_list_filter(title_segment_list)
        for title_segment in title_segment_list:
            keywords_dic[title_segment] = len(keywords_dic)
        
        content_segment_list_ = []
        
        for content_segment in content_segment_list:
            if content_segment.strip() != "":
                if content_segment not in keywords_dic.keys():
                    content_segment_list_.append(content_segment)
                else:
                    keywords = "keywords" + str(keywords_dic[content_segment])
                    content_segment_list_.append(keywords)
            
        return content_segment_list_ 
    
    def hanlp_rough_segment(self, content):
    
        hanlp_rough_url = 'http://hanlp-rough-service:31001/hanlp/segment/rough?'
        
        params = {'content':content}
        
        response = self.requests_post(hanlp_rough_url, params)
        data = response['data']
        
        segments = []
        for x in data:
            segments.append(x['word'])
            
        return segments
    
    # data = {}
    def requests_post(self, url, data):
        
        # 注意：
        #   data 可以是 dic，也可以是 list 等
        #data = json.dumps(params, ensure_ascii = False, indent = 2)
        data = json.dumps(data)
        #data = json.dumps(data).encode("UTF-8")
        response = requests.post(url, data = data)
        response = response.json()
        
        return response

# hot spot description and influence/comment/point of view
# special_signs = ['~~[[1', '1]]~~', '~~[[2', '2]]~~']
class Service_Dataset_HotSpot_title_encoding_artificial(object):
    """Class that iterates over Dataset

    __iter__ method yields a tuple (words, label)
        words: list of raw words
        label: label

    If processing_word and processing_tag are not None,
    optional preprocessing is appplied

    Example:
        ```python
        data = Dataset(filename)
        for sentence, labels in data:
            pass
        ```

    """
    def __init__(self, data, processing_x=None,
                 max_iter=None):
        """
        Args:
            filename: path to the file
            processing_x: (optional) function that takes a word as input
            processing_y: (optional) function that takes a label as input
            max_iter: (optional) max number of sentences to yield
        """
        self.data = data
        self.processing_x = processing_x
        self.max_iter = max_iter
        self.length = None
        
    # iterator
    def __iter__(self):
        '''
        e.g.
        yield 每条数据----> 
        (['word1', 'word2', 'word3', 'word4', 'word5', 'word6'], 1.0)
        (['word1', 'word2', 'word3', 'word4', 'word5', 'word6'], [0, 1])
        (['word1', 'word2', 'word3', 'word4', 'word5', 'word6'], 8)
        '''
        # 当前在 iterator 中的 sentences 个数
        niter = 0
            
        for dic in self.data:
            
            title = dic['title']
            content = dic['content']
            
            title_segments = self.hanlp_rough_segment(title)
            content_segments = self.hanlp_rough_segment(content)
            
            x  = self.title_encoding_artificial(title_segments, content_segments)
        
            # 一条数据
            words = []
            for i in range(len(x)):
                
                # x
                word = x[i]
                # strio() 为构建字典不重复，与 load_vocab 相呼应
                word = word.strip()
                
                if self.processing_x is not None:
                    word = self.processing_x(word)
                    
                words += [word]
            
            yield words
                
            niter += 1
            if self.max_iter is not None and niter > self.max_iter:
                break

    # 重写了 len(iterator)
    def __len__(self):
        """Iterates once over the corpus to set and store length"""
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1

        return self.length
     
    # 分词后 的 title_segment_list 去掉了 “ ”
    def title_segment_list_filter(self, title_segment_list):
        
        title_segment_list_ = []
        stop_words = ['的', '：','！']
        for title_segment in title_segment_list:
            if title_segment not in stop_words and title_segment.strip() != "":
                title_segment_list_.append(title_segment)
        
        return title_segment_list_
    
    # 分词后 的 content_segment_list 去掉了 “ ”
    def title_encoding_artificial(self, title_segment_list, content_segment_list):
        
        keywords_dic = {}
        title_segment_list = self.title_segment_list_filter(title_segment_list)
        for title_segment in title_segment_list:
            keywords_dic[title_segment] = len(keywords_dic)
        
        content_segment_list_ = []
        
        for content_segment in content_segment_list:
            if content_segment.strip() != "":
                if content_segment not in keywords_dic.keys():
                    content_segment_list_.append(content_segment)
                else:
                    keywords = "keywords" + str(keywords_dic[content_segment])
                    content_segment_list_.append(keywords)
            
        return content_segment_list_ 
    
    def hanlp_rough_segment(self, content):
    
        hanlp_rough_url = 'http://hanlp-rough-service:31001/hanlp/segment/rough?'
        
        params = {'content':content}
        segments = []
        
        try:
            response = self.requests_post(hanlp_rough_url, params)
            data = response['data']

            for x in data:
                segments.append(x['word'])
        except Exception as e:
            logging.exception(e)
            logging.exception("hanlpSegmentError" + content)
            
        return segments
    
    # data = {}
    def requests_post(self, url, data):
        
        # 注意：
        #   data 可以是 dic，也可以是 list 等
        #data = json.dumps(params, ensure_ascii = False, indent = 2)
        data = json.dumps(data)
        #data = json.dumps(data).encode("UTF-8")
        response = requests.post(url, data = data)
        response = response.json()
        
        return response

    
def get_vocabs(dataset):
    """Build vocabulary from an iterable of datasets objects

    Args:
        datasets: a list of dataset objects

    Returns:
        a set of all the words in the dataset

    """
    print("Building vocab...")
    
    vocab_words = set()
    vocab_labels = set()
        
    for words, label in dataset:
        vocab_words.update(words)
        if type(label) == list:
            pass
        else:
            vocab_labels.update([label])
        
    print("- done. {} tokens".format(len(vocab_words)))
    print("- done. {} tokens".format(len(vocab_labels)))
    
    return vocab_words, vocab_labels

def write_vocab(vocab, filename):
    """Writes a vocab to a file

    Writes one word per line.

    Args:
        vocab: iterable that yields word
        filename: path to vocab file

    Returns:
        write a word per line

    """
    print("Writing vocab...")
    with open(filename, "w") as f:
        for i, word in enumerate(vocab):
            if i != len(vocab) - 1:
                f.write("{}\n".format(word))
            else:
                f.write(word)
    print("- done. {} tokens".format(len(vocab)))

def load_vocab(filename):
    """Loads vocab from a file

    Args:
        filename: (string) the format of the file must be one word per line.

    Returns:
        d: dict[word] = index

    """
    try:
        d = dict()
        with open(filename, encoding = 'utf-8') as f:
            for idx, word in enumerate(f):
                word = word.replace('\n','')
                d[word] = idx

    except IOError:
        raise Exception("No vocab path:  "+ filename)
        
    return d

def get_processing(vocab_words=None, 
                      lowercase=False, 
                          allow_unk=False):
    """Return lambda function that transform a word (string) into list,
    or tuple of (list, id) of int corresponding to the ids of the word and
    its corresponding characters.

    Args:
        vocab: dict[word] = idx

    Returns:
        f("word") = word id
    """
    def f(word):

        # preprocess word
        if lowercase:
            word = word.lower()
        
        # 需要考虑，暂时不使用
        #if word.isdigit():
        #    word = NUM
        
        # get id of word
        if vocab_words is not None:
            if word in vocab_words:
                word = vocab_words[word]
            else:
                if allow_unk:
                    word = vocab_words[UNK]
                else:
                    # 若果不加 UNK，train 和 prediction 可能会出现未知字符
                    # 原因在于 对 word.strip() 的处理 肯能会有问题
                    print('word length: '+ str(len(word)))
                    print('word: '+ str(len(word)))
                    raise Exception("Unknow key is not allowed. Check that "\
                                    "your vocab (tags?) is correct")

        return word

    return f


def _pad_sequences(sequences, pad_tok, max_length):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with

    Returns:
        a list of list where each sublist has same length
    """
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok]*max(max_length - len(seq), 0)
        sequence_padded +=  [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


def pad_sequences(sequences, pad_tok, max_length=None, nlevels=1):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
        nlevels: "depth" of padding, for the case where we have characters ids

    Returns:
        a list of list where each sublist has same length

    """
    if nlevels == 1:
        if max_length==None:
            max_length = max(map(lambda x : len(x), sequences))
        else:
            max_length = max_length 
        sequence_padded, sequence_length = _pad_sequences(sequences,
                                            pad_tok, max_length)

    elif nlevels == 2:
        max_length_word = max([max(map(lambda x: len(x), seq))
                               for seq in sequences])
        sequence_padded, sequence_length = [], []
        for seq in sequences:
            # all words are same length now
            sp, sl = _pad_sequences(seq, pad_tok, max_length_word)
            sequence_padded += [sp]
            sequence_length += [sl]

        max_length_sentence = max(map(lambda x : len(x), sequences))
        sequence_padded, _ = _pad_sequences(sequence_padded,
                [pad_tok]*max_length_word, max_length_sentence)
        sequence_length, _ = _pad_sequences(sequence_length, 0,
                max_length_sentence)

    return sequence_padded, sequence_length

def minibatches(data, minibatch_size, shuffle = False):
    """
    Args:
        data: generator of (sentence, tags) tuples
        minibatch_size: (int)

    Yields:
        list of tuples

    """
    if shuffle:
        x_ = []
        y_ = []
        for (x, y) in data:
            x_ += [x]
            y_ += [y]
        
        np.random.seed(10)
        shuffle_indices = np.random.permutation(np.arange(len(y_)))
        x_ = np.array(x_)
        y_ = np.array(y_)
        x_ = x_[shuffle_indices]
        y_ = y_[shuffle_indices]
        
        x_batch, y_batch = [], []
        for (x, y) in zip(x_, y_):
            # x = [(char_list, ansj_tag, word), () ... ()]
            if len(x_batch) == minibatch_size:
                yield x_batch, y_batch
                x_batch, y_batch = [], []
    
            x_batch += [x]
            y_batch += [y]
        
        # 最后一个循环时需要多做处理
        if len(x_batch) != 0:
            yield x_batch, y_batch
        
    else:
        
        x_batch, y_batch = [], []
        for (x, y) in data:
            # x = [(char_list, ansj_tag, word), () ... ()]
            if len(x_batch) == minibatch_size:
                yield x_batch, y_batch
                x_batch, y_batch = [], []
    
            x_batch += [x]
            y_batch += [y]
        
        # 最后一个循环时需要多做处理
        if len(x_batch) != 0:
            yield x_batch, y_batch

def service_predict_minibatches(data, minibatch_size):
    """
    Args:
        data: generator of sentence tuples
        minibatch_size: (int)

    Yields:
        list of tuples

    """
    x_batch = []
    for x in data:
        # x = [(char_list, ansj_tag, word), () ... ()]
        if len(x_batch) == minibatch_size:
            yield x_batch
            x_batch = []
            
        x_batch += [x]
    
    # 最后一个循环时需要多做处理
    if len(x_batch) != 0:
        yield x_batch
