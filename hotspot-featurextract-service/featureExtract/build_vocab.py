# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

from config import Config
from data_helper import UNK, Dataset_HotSpot_title_encoding_artificial, NUM
from data_helper import get_vocabs, write_vocab,load_vocab
import gensim

def main():
    """Procedure to build data

    You MUST RUN this procedure. It iterates over the whole dataset (train,
    dev and test) and extract the vocabularies in terms of words, tags, and
    characters. Having built the vocabularies it writes them in a file. The
    writing of vocabulary in a file assigns an id (the line #) to each word.
    It then extract the relevant GloVe vectors and stores them in a np array
    such that the i-th entry corresponds to the i-th word in the vocabulary.


    Args:
        config: (instance of Config) has attributes like hyper-params...

    """
    
    # get config and processing of words
    config = Config(load=False)
    
    # Generators
    train = Dataset_HotSpot_title_encoding_artificial(config.filename_train)

    # Build Word and Tag vocab
    # 小写， 半角
    vocab_words, _ = get_vocabs(train)
    
    if config.embeddings is not None:
        # 加载 gensim words vocab
        model = gensim.models.Word2Vec.load(config.w2v_words)
        vocab_words = []
        vocab_words.append(UNK)
        #vocab_words.append(NUM)
        for word in model.wv.vocab.keys():
            vocab_words.append(word)
    else:
        vocab_words_ = []
        vocab_words_.append(UNK)
        #vocab_words.append(NUM)
        for i, word in enumerate(vocab_words):
            vocab_words_.append(word)
        vocab_words = vocab_words_

    # Save vocab
    write_vocab(vocab_words, config.vocab_words_path)
    
    # 验证 write_vocab 与 load_vocab 是否一致
    d = load_vocab(config.vocab_words_path)
    if len(vocab_words) == len(d.keys()):
        print('write_vocab and load_vocab cool!')
        
        try:
            print('UNK ids:', d[UNK])
        except:
            raise Exception("Check UNK in vocab_words and load_vocab")
        
    else:
        raise Exception("Check vocab_words and load_vocab")
        
    
if __name__ == "__main__":
    
    main()

