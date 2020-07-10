# -*- coding: utf-8 -*-
import requests
import json
import random

def load_json(file_path):
    
    with open(file_path, 'r', encoding = 'utf-8') as f1:
        data = json.load(f1) 
   
    return data

# 将字典写入文件
def dump_json_to_file(my_dic, file_path):
    
    jsObj = json.dumps(my_dic, ensure_ascii = False, indent = 2)  
    fileObject = open(file_path, 'w', encoding = 'utf-8')  
    fileObject.write(jsObj)  
    fileObject.close() 
    
# params = {}
def requests_get(url, params):

    response = requests.get(url, params = params)
    response = response.json()
    
    return response

# 分词后 的 title_segment_list 去掉了 “ ”
def title_segment_list_filter(title_segment_list):
    
    title_segment_list_ = []
    stop_words = ['的', '：','！']
    for title_segment in title_segment_list:
        if title_segment not in stop_words and title_segment.strip() != "":
            title_segment_list_.append(title_segment)
    
    return title_segment_list_

# 分词后 的 content_segment_list 去掉了 “ ”
def title_encoding_artificial(title_segment_list, content_segment_list):
    
    keywords_dic = {}
    title_segment_list = title_segment_list_filter(title_segment_list)
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

def hanlp_rough_segment(content):
    
    hanlp_rough_url = 'http://hanlp-rough-service:31001/hanlp/segment/rough?'
    
    params = {'content':content}
    
    response = requests_get(hanlp_rough_url, params)
    data = response['data']
    
    segments = []
    for x in data:
        segments.append(x['word'])
        
    return segments
    
if __name__ == "__main__":
    
    '''
    title_content = "碳纳米新型电池正在研制  电动汽车或将迎电池革命"
    title_segments = hanlp_rough_segment(title_content)
    
    content = "法国创业公司 Nawa  Technologies 官方表示，公司正在研发一种新型电池，这种电池在融入公司的核心产品------新型碳纳米超级电容器后，能够在短短数秒中完成汽车充电，同时重量也有明显下降。由于没有发生化学反应，仅仅只是质子和离子之间的物理分离，超快充电并不会导致电池产生热量或者膨胀。这意味着碳纳米超级电容器的使用寿命非常长，充电周期可以高达 100 万次。 "
    content_segments = hanlp_rough_segment(content)
    
    content_segment_list_ = title_encoding_artificial(title_segments, content_segments)
    
    print(content_segment_list_)
    '''
    data = load_json('./data/data.json')
    
    max_len = 0
    train_ = {}
    dev_ = {}
    
    r = list(range(1,len(data)+1))
    random.shuffle(r) 
    
    a = 1
    b = 1
    c = 1
    
    n = 1
    m = 1
    
    for i in r:
        
        temp_dic = {}
        
        title = data[str(i)]['title']
        content = data[str(i)]['content']
        title_segments = hanlp_rough_segment(title)
        content_segments = hanlp_rough_segment(content)
        
        if len(content_segments) > max_len:
            max_len = len(content_segments)
        
        temp_dic['title_segments'] = title_segments
        temp_dic['content_segments'] = content_segments
        temp_dic['label'] = data[str(i)]['label']
        
        if data[str(i)]['label'] == '概述' and a < 4:
            
            a += 1
            dev_[n] = temp_dic
            n += 1
            continue
        
        if data[str(i)]['label'] == '观点' and b < 4:
           
            b += 1
            dev_[n] = temp_dic
            n += 1
            continue
        
        if data[str(i)]['label'] == '其它' and c < 78:
            
            c += 1
            dev_[n] = temp_dic
            n += 1
            continue
        
        train_[m] = temp_dic
        m += 1
             
    print(max_len)
    dump_json_to_file(train_, './data/train_.json')
    dump_json_to_file(dev_ ,'./data/dev_.json')
    
    
    




