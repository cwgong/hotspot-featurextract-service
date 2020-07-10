# -*- coding: utf-8 -*-

import sys
sys.path.append('./featureExtract')

# 自定义版本
from _version import __version__

# python
import io
import json

# tornado
import logging
import tornado.escape
import tornado.ioloop
import tornado.web
import tornado.httpserver
import tornado.options
from tornado.escape import json_decode

# business logical
from featureExtract.cnn_model import CNN_Model
from featureExtract.config import Config
from featureExtract.data_helper import Service_Dataset_HotSpot_title_encoding_artificial
import datetime
import re
import requests
import codecs

# const
# version
VERSION = "0.7"
# 定义常量
NEG_VIEWS_KEYWORDS = ['下称','持股','公告称','先后发言','发言人','发言','会上','发了言','习近平指出','习近平强调','习近平在讲话中','习近平曾经说','回应称','会议指出','会议强调','会议称','致辞','说清楚','可以说','所以说','是否认为']
NEG_NR = ['李克强','习近平','彭华岗','关键时','危为机','常观']
NEG_FXS = ['李克强','彭博','易纲','方星海','孟玮']
VIEWS_KEYWORDS = ["表示","认为","称","指出","说","强调","看来","介绍","点评","发表","提到","体现","意味","意味着","来看","观点","我们","事实上","专家","而言","提出","分析"]
SPECIAL_IDENTITIES = ['证券部人士', '投行人士', '投行业内人士', '创投界人士', '市场分析人士','私募分析人士','专家认为','资深专家','从业人士','资深专家及从业人士','业内人士']
SECURITIES = []

def read_file(filepath):
    file = codecs.open(filepath, encoding = 'utf-8')
    lines = file.readlines()
    return lines

from functools import wraps
 
def singleton(cls):
    instances = {}
    @wraps(cls)
    def getinstance(*args, **kw):
        if cls not in instances:
            instances[cls] = cls(*args, **kw)
        return instances[cls]
    return getinstance

@singleton
class tf_model_config(Config):
    
    def __init__(self):
        
        Config.__init__(self)
        logging.info("tf_model_config inititial ... ")

@singleton       
class tf_model(CNN_Model):
    
    def __init__(self, config):
        
        CNN_Model.__init__(self, config)
        logging.info("tf_model inititial ... ")
        
# config
def parse_conf_file(config_file):
    
    config = {}
    with io.open(config_file, 'r', encoding='utf8') as f:
        config = json.load(f)
    return config  

class PredictHandler(tornado.web.RequestHandler):      

    def get(self):  

        try:
            logging.info("get")
            begin = datetime.datetime.now()
            
            list_of_doc = self.get_argument("list_of_doc")
            list_of_doc = json_decode(list_of_doc)
            logging.info("\n提取文章个数: ---------> " + str(len(list_of_doc)))
            
            # predict...
            self.handler_predict_rule(list_of_doc)
            
            end = datetime.datetime.now()
            time = end - begin
            logging.info("\nget success! ------> " + "  time: " + str(time))
        except Exception as e:
            logging.exception(e)
    
    def post(self):

        try:
            logging.info("post")
            begin = datetime.datetime.now()
            
            body_data = json_decode(self.request.body)
            list_of_doc = body_data
            logging.info("\n提取文章个数: ---------> " + str(len(list_of_doc)))
            
            self.handler_predict_rule(list_of_doc)

            end = datetime.datetime.now()
            time = end - begin
            logging.info("\nget success! ------> " + "  time: " + str(time))
        except Exception as e:
            logging.exception(e)

    
    def handler_predict_rule(self, list_of_doc):

        def hanlp_segment(content):            
            hanlp_url = "http://hanlp-rough-service:31001/hanlp/segment/rough?"
            data = {'content':content}
            response = requests_post(hanlp_url, data)
            data = response['data']   
            return data
        def requests_post(url, data):
        
            data = json.dumps(data)
            #data = json.dumps(data).encode("UTF-8")
            response = requests.post(url, data = data, headers={'Connection':'close'})
            response = response.json()
            return response
        def removeAllTag(s):
            s = re.sub('<[^>]+>','',s)
            return s
        def gs_rule(keywords, p):
            if len(p) > 40 and len(p) < 500:
                count = 0
                for word in keywords:
                    if word in p:
                        count +=1
                if count >= int(len(keywords)*0.7):
                    if '月' in p and '日' in p:
                        return 1
                    if '昨日' in p or '今日' in p:
                        return 1
                    if '近日' in p:
                        return 1
                    if '日' in p:
                        return 1
            return 0
        def abstract_rule(p):
            if len(p) > 40 and len(p) < 500:
                if '月' in p and '日' in p:
                    if '摄' not in p:
                        return 1
                if '昨日' in p or '今日' in p:
                    return 1
                if '近日' in p:
                    return 1
                if '日' in p:
                    return 1
            return 0
        def abstract_score(p):
            if '月' in p and '日' in p and '电' in p:
                score = 0.81
                return score
            if '月' in p and '日' in p and '消息' in p:
                score = 0.81
                return score
            if '月' in p and '日' in p:
                score = 0.8
                return score
            if '近日' in p:
                score = 0.6
                return score
            if '日' in p:
                score = 0.6
                return score

            return 0.5
        # 通过 title 判断是否为评论性文章，并给出作者或机构
        def critical_report_judge(title, keywords):
            # '''
            # 根据观察的数据特点，根据标题判断是否为评论性文章
            # 1、包含关键词（相对宽松）
            # 2、包含券商机构或分析师
            # '''
            author = ''
            author_type = ''
            if "：" in title:
                count = 0
                for word in keywords:
                    if word in title:
                        count +=1
                if count >= int(len(keywords)*0.5):
                    try:
                        terms = hanlp_segment(title)
                    except Exception as e:
                        return 0, author, author_type
                    # 先找 作者
                    for term in terms:
                        if term['nature'] == 'fxs' or term['word'] == '巴曙松':
                            if term['word'] not in NEG_FXS:
                                author = term['word']
                                author_type = 'fxs'
                                logging.info("critical_report_judge author： " + author) 
                                logging.info("critical_report_judge author_type： " + author_type) 
                                return 1, author, author_type
                    # 再找 券商  
                    for security in SECURITIES:
                        if security in title:
                            author = security
                            author_type = 'security'
                            logging.info("critical_report_judge author： " + author) 
                            logging.info("critical_report_judge author_type： " + author_type) 
                            return 1, author, author_type
            return 0, author, author_type
        def critical_gd_rule(keywords, p):
            logging.info("critical_gd_rule p： " + p) 
            if len(p) > 40 and len(p) < 500 and "记者" not in p:
                
                # 还是需要先分词，因为关键词中的 “称”、“说”
                try:
                    terms = hanlp_segment(p)
                except Exception as e:
                    logging.info("hanlp_segment error： " + p) 
                    return 0
                
                p_segs = [x['word'] for x in terms]
                
                c1 = 0
                for word in VIEWS_KEYWORDS:
                    if word in p_segs:
                        neg = 0
                        for word1 in NEG_VIEWS_KEYWORDS:
                            if word1 in p:
                                neg +=1
                                break
                        if neg == 0:
                            c1+=1
                            break
                logging.info("c1： " + str(c1)) 
                c2 = 0
                for word in keywords:
                    if word in p:
                        c2 +=1
                logging.info("c2： " + str(c2)) 
                if c1 != 0 and c2 >= int(len(keywords)*0.5):
                    return 1
            return 0
        def gd_rule(keywords, p):
            logging.info("gd_rule p： " + p) 
            author = ''
            author_type = ''
            if len(p) > 40 and len(p) < 500 and "记者" not in p:
                
                # 还是需要先分词，因为关键词中的 “称”、“说”
                try:
                    terms = hanlp_segment(p)
                except Exception as e:
                    logging.info("hanlp_segment error： " + p) 
                    return 0, author, author_type
                
                p_segs = [x['word'] for x in terms]
                
                c1 = 0
                for word in VIEWS_KEYWORDS:
                    if word in p_segs:
                        neg = 0
                        for word1 in NEG_VIEWS_KEYWORDS:
                            if word1 in p:
                                neg +=1
                                break
                        if neg == 0:
                            c1+=1
                            break
                logging.info("c1： " + str(c1)) 
                c2 = 0
                for word in keywords:
                    if word in p:
                        c2 +=1
                logging.info("c2： " + str(c2)) 
                if c1 != 0 and c2 >= int(len(keywords)*0.5):
                    
                    # 先找 作者
                    for term in terms:
                        if term['nature'] == 'fxs' or term['word'] == '巴曙松':
                            if term['word'] not in NEG_FXS:
                                author = term['word']
                                author_type = 'fxs'
                                logging.info("gd_rule author： " + author)
                                logging.info("gd_rule author_type： " + author_type)
                                return 1, author, author_type
                    # 再找特殊身份人士 如 “投行人士”
                    for special_identity in SPECIAL_IDENTITIES:
                        if special_identity in p:
                            author = special_identity
                            author_type = 'special_identity'
                            if author == '专家认为':
                                logging.info("gd_rule author： " + '专家') 
                                logging.info("gd_rule author_type： " + author_type)
                                return 1, '专家', author_type
                            logging.info("gd_rule author： " + author) 
                            return 1, author, author_type
                    # 再找 券商  
                    for security in SECURITIES:
                        if security in p:
                            author = security
                            author_type = 'security'
                            logging.info("gd_rule author： " + author) 
                            logging.info("gd_rule author_type： " + author_type)
                            return 1, author, author_type
            return 0, author, author_type
        
        # 打分
        def gd_score(p, keywords, author_type, article_type):
            
            sum_keywords_score = len(keywords) * 3
            keywords_score = 0
            for keyword in keywords:
                if keyword in p:
                    keywords_score += 3
            
            author_type_score = 0
            sum_author_type_score = 20
            if author_type in ['fxs', 'special_identity']:
                author_type_score = 20
            elif author_type == 'security':
                author_type_score = 15
            elif author_type == 'nr':
                author_type_score = 5
                
            article_type_score = 0
            sum_article_type_score = 5
            if article_type == 1:
                article_type_score = 5
            
            # sum_score = 15 + 20 + 15
            score = (keywords_score + author_type_score + article_type_score)/(sum_keywords_score + sum_author_type_score +sum_article_type_score)
            score = score/2.5 + 0.6
            
            if author_type == 'nr' or author_type == '':
                score = (keywords_score + author_type_score + article_type_score)/(sum_keywords_score + sum_author_type_score +sum_article_type_score)
            
            return score
        
        # 打分
        def gs_score(p, keywords):
            
            sum_keywords_score = len(keywords) * 2
            keywords_score = 0
            for keyword in keywords:
                if keyword in p:
                    keywords_score += 2
            
            score = keywords_score/sum_keywords_score
            score = score/5 -0.1
            if '消息' in p:
                score += 0.1
            score = score + 0.8
            
            return score
                
        
        # 条件不严格的点评, 可不包含事件的关键词
        def generalized_gd_rule(p):
            logging.info("generalized_gd_rule p： " + p) 
            author = ''
            author_type = ''
            if len(p) > 40 and len(p) < 500 and "记者" not in p:
                
                # 还是需要先分词，因为关键词中的 “称”、“说”
                try:
                    terms = hanlp_segment(p)
                except Exception as e:
                    logging.info("hanlp_segment error： " + p) 
                    return 0, author, author_type
                
                p_segs = [x['word'] for x in terms]
                
                c1 = 0
                if '分析认为' in p:
                    c1 += 1
                
                for word in VIEWS_KEYWORDS:
                    if word in p_segs:
                        neg = 0
                        for word1 in NEG_VIEWS_KEYWORDS:
                            if word1 in p:
                                neg +=1
                                break
                        if neg == 0:
                            c1+=1
                            break
                logging.info("c1： " + str(c1)) 
                
                if c1 != 0:
                    try:
                        terms = hanlp_segment(p)
                    except Exception as e:
                        logging.info("hanlp_segment error： " + p) 
                        return 0, author, author_type
                    
                    # 先找特殊身份人士 如 “投行人士”
                    for special_identity in SPECIAL_IDENTITIES:
                        if special_identity in p:
                            author = special_identity
                            author_type = 'special_identity'
                            if author == '专家认为':
                                logging.info("generalized_gd_rule author： " + '专家') 
                                return 1, '专家', author_type
                            logging.info("generalized_gd_rule author： " + author) 
                            return 1, author, author_type
                    # 再找 券商  
                    for security in SECURITIES:
                        if security in p:
                            author = security
                            author_type = 'security'
                            logging.info("generalized_gd_rule author： " + author) 
                            return 1, author, author_type
                    # 再找 nr
                    for term in terms:
                        if term['nature'] == 'nr':
                            if term['word'] not in NEG_NR:
                                author = term['word']
                                author_type = 'nr'
                                logging.info("generalized_gd_rule author： " + author) 
                                return 1, author, author_type
                            
                    # 最后符合表述的句式就可以
                    # 噪音太大
                    return 1, author, author_type
                    
                    
            return 0, author, author_type
        
        
        new_list_of_doc = []
        preds = []
        
        for doc in list_of_doc:
            try:
                id = doc['id']
                logging.info('doc_id: ' + str(id))
                title = doc['title']
                logging.info('title: ' + str(title))
                content = doc['content']
                keywords = doc['keywords']
                logging.info('keywords: ' + str(keywords))
                cluster_title = ''
                if 'clusterTitle' in doc.keys():
                    cluster_title = doc['clusterTitle']
                    logging.info('clusterTitle: ' + str(cluster_title))
                if 'clusterId' in doc.keys():
                    clusterId = doc['clusterId']
                    logging.info('clusterId: ' + str(clusterId))
                if 'summary' in doc.keys():
                    summary = doc['summary']
                else:
                    summary = ''
                if doc["dataSource"] == "CRAWL":
                    content = content.split('</p><p>')  
                else:
                    content = content.split('</p></p><p><p>')
            except Exception as e:
                logging.info(e)
                logging.info('get id_title_content_summary_source from yh error!')
                continue
            
            # ======================
            # Summary 处理
            # ======================
            preds_per = []
            # 一个 doc 只有一个 概述
            gs_condition = 0
            # 如果爬取到了“概述”
            if len(summary) !=0:
                summary_ = removeAllTag(summary)
                if gs_rule(keywords, summary_) == 1:
                    logging.info('abstract: ' + summary_)
                    score = gs_score(summary_, keywords)
                    temp_dic = {}
                    temp_dic['id'] = id
                    temp_dic['content'] = summary_
                    temp_dic['label'] = "事件概述"
                    temp_dic['relationScore'] = score
                    new_list_of_doc.append(temp_dic)
                    gs_condition += 1
                    preds_per.append(1)
                    

            # 处理掉 异常 文章，段落特多的
            if len(content) < 100:
                
                # ======================
                # 概述抽取
                # ======================
                # 每个文章的每一段作处理 抽取概述
                if gs_condition == 0:
                    for i in range(0, len(content)):
                        p = removeAllTag(content[i])
                        if i < 6:
                            if gs_rule(keywords, p) == 1:
                                logging.info('abstract_: ' + p)
                                score = gs_score(p, keywords)
                                temp_dic = {}
                                temp_dic['id'] = id
                                temp_dic['content'] = p
                                temp_dic['label'] = "事件概述"
                                temp_dic['relationScore'] = score
                                new_list_of_doc.append(temp_dic)
                                gs_condition += 1
                                preds_per.append(2)
                                break
                
                # 为了解决目前簇内必须有和簇内标题一致概述的 bug，与簇标题一致的文章必须抽取出概述，即使不和簇内的 keywords 有关，或可能是簇内的杂质。
                if gs_condition == 0 and cluster_title == title:
                    for i in range(0, len(content)):
                        p = removeAllTag(content[i])
                        if i < 6:
                            if abstract_rule(p) == 1:
                                logging.info('abstract1_: ' + p)
                                score = abstract_score(p)
                                temp_dic = {}
                                temp_dic['id'] = id
                                temp_dic['content'] = p
                                temp_dic['label'] = "事件概述"
                                temp_dic['relationScore'] = score
                                new_list_of_doc.append(temp_dic)
                                gs_condition += 1
                                preds_per.append(22)
                                break
                
                if gs_condition == 0 and cluster_title == title:
                    for i in range(0, len(content)):
                        p = removeAllTag(content[i])
                        if i < 6:
                            if len(p) > 20 and len(p) < 500:
                                logging.info('abstract2_: ' + p)
                                score = abstract_score(p)
                                temp_dic = {}
                                temp_dic['id'] = id
                                temp_dic['content'] = p
                                temp_dic['label'] = "事件概述"
                                temp_dic['relationScore'] = score
                                new_list_of_doc.append(temp_dic)
                                gs_condition += 1
                                preds_per.append(222)
                                break
                            
                # 继续补漏
                if gs_condition == 0 and cluster_title == title:
                    p = '\r\n'.join([removeAllTag(cont) for cont in content[0:5]])    
                    logging.info('abstract3_: ' + p)
                    score = abstract_score(p)
                    temp_dic = {}
                    temp_dic['id'] = id
                    temp_dic['content'] = p
                    temp_dic['label'] = "事件概述"
                    temp_dic['relationScore'] = score
                    new_list_of_doc.append(temp_dic)
                    gs_condition += 1
                    preds_per.append(2222)
                
                # ======================
                # 观点抽取
                # ======================
                # 判断文章类型（是否为评论性文章）
                crj, author, author_type = critical_report_judge(title, keywords) 
                if crj == 1:
                # 评论性文章
                    for i in range(0, len(content)):
                        p = removeAllTag(content[i])
                        cgr = critical_gd_rule(keywords, p)
                        if cgr == 1:
                            logging.info('view: ' + p)
                            score = gd_score(p, keywords, author_type, 1)
                            logging.info('score: ' +  str(score))
                            temp_dic = {}
                            temp_dic['id'] = id
                            temp_dic['content'] = p
                            temp_dic['label'] = "事件影响"
                            temp_dic['author'] = author
                            temp_dic['relationScore'] = score
                            new_list_of_doc.append(temp_dic)
                            preds_per.append(3)
                else:
                # 非评论性文章
                    # 严格
                    for i in range(0, len(content)):
                        p = removeAllTag(content[i])
                        gr, author, author_type = gd_rule(keywords, p)
                        if gr == 1:
                            logging.info('view: ' + p)
                            score = gd_score(p, keywords, author_type, 0)
                            logging.info('score: ' + str(score))
                            temp_dic = {}
                            temp_dic['id'] = id
                            temp_dic['content'] = p
                            temp_dic['label'] = "事件影响"
                            temp_dic['author'] = author
                            temp_dic['relationScore'] = score
                            new_list_of_doc.append(temp_dic)
                            preds_per.append(4)
                        else:
                            # 避免概述    
                            if i > 3:
                                # 不严格
                                ggr, author, author_type = generalized_gd_rule(p)
                                if ggr == 1:
                                    logging.info('view: ' + p)
                                    score = gd_score(p, keywords, author_type, 0)
                                    logging.info('score: ' + str(score))
                                    temp_dic = {}
                                    temp_dic['id'] = id
                                    temp_dic['content'] = p
                                    temp_dic['label'] = "事件影响"
                                    temp_dic['author'] = author
                                    temp_dic['relationScore'] = score
                                    new_list_of_doc.append(temp_dic)
                                    preds_per.append(5)
                        
            else:
                logging.info('this doc content is too large!')
                if gs_condition == 0 and cluster_title == title:
                    p = '\r\n'.join([removeAllTag(cont) for cont in content[0:5]])    
                    logging.info('abstract2_: ' + p)
                    score = abstract_score(p)
                    temp_dic = {}
                    temp_dic['id'] = id
                    temp_dic['content'] = p
                    temp_dic['label'] = "事件概述"
                    temp_dic['relationScore'] = score
                    new_list_of_doc.append(temp_dic)
                    gs_condition += 1
                    preds_per.append(222)
                    
            preds.append(preds_per)
        
        logging.info('preds_per in preds: ' + str(preds))

        result_json = json.dumps(new_list_of_doc, ensure_ascii=False)
        self.write(result_json) 
        logging.info('writ result json finisehd!')
       
    
class Application(tornado.web.Application):
    
    def __init__(self, config):
        handlers = [
            (config['url'], PredictHandler), 
        ]
        settings = dict(
            debug = bool(config['debug']),
        )
        tornado.web.Application.__init__(self, handlers, **settings)

        
def main(argv):
    
    if sys.version_info < (3,):
        # reload(sys)
        sys.setdefaultencoding("utf-8")
   
    if VERSION != __version__:
        print("version error!")
        logging.info("version error!")
        exit(-1)
    
    if len(argv) < 2:
        print('arg error.')
        exit(-2)  
        
    config = parse_conf_file(argv[1])
    tornado.options.parse_config_file(config['log_config_file'])
    
    logging.info("Server Inititial ... ")
    
    # 由于目前数据变动较大，且训练数据太少，模型预测效果不佳，故先不经过模型
    # initial model
    #modelConfig = tf_model_config()
    #featureExtractModel = tf_model(modelConfig)
    #featureExtractModel.build()
    #featureExtractModel.restore_session(modelConfig.dir_model)
    #logging.info("tf_model inititial success! ")
    
    securities_name = read_file('securities_name.txt')
    for x in securities_name:
        SECURITIES.append(x.strip())
    logging.info('所有券商个数： '+ str(len(SECURITIES)))
    
    app = Application(config)
    server = tornado.httpserver.HTTPServer(app)
    server.bind(config['port'])
    server.start(config['process_num'])
    logging.info("Server Inititial Success! ")
    print("Server Inititial Success! ")
    tornado.ioloop.IOLoop.current().start()

if __name__ == "__main__":
    
    main(sys.argv)