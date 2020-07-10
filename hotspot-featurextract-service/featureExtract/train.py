# -*- coding: utf-8 -*-

from data_helper import Dataset_HotSpot_title_encoding_artificial
from cnn_model import CNN_Model
from config import Config
from threading import Timer

def main():
    
    config = Config()
    
    train = Dataset_HotSpot_title_encoding_artificial(config.filename_train, config.processing_x, max_iter = config.max_iter)
    dev = Dataset_HotSpot_title_encoding_artificial(config.filename_dev, config.processing_x)
    
    config.logger.info('train ----> ' + str(len(train)) +' dev ----> ' + str(len(dev)))
    
    
    # build model
    model = CNN_Model(config)
    model.build()
    
    model.train(train, dev)
    

if __name__ == '__main__':
    
    '''
    t = Timer(3*60*60, main)
    t.start()
    '''
    main()
    
    

        
    
    
    
    
    
    
    
    
    
    
    