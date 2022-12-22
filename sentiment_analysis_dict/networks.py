# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 11:48:24 2017

@author: cm
"""

import os
import sys
import jieba
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from sentiment_analysis_dict.utils import ToolGeneral
from sentiment_analysis_dict.hyperparameters import Hyperparams as hp


tool = ToolGeneral()
jieba.load_userdict(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dict','jieba_sentiment.txt'))


class SentimentAnalysis():
    """
    Sentiment Analysis with some dictionarys
    """      
    def sentiment_score_list(self,dataset):
        seg_sentence = tool.sentence_split_regex(dataset)
        count1,count2 = [],[]
        for sentence in seg_sentence: 
            words = jieba.lcut(sentence, cut_all=False)
            i = 0 
            a = 0 
            for word in words:
                """
                poscount 积极词的第一次分值;
                poscount2 积极反转后的分值;
                poscount3 积极词的最后分值（包括叹号的分值）      
                """
                poscount,negcount,poscount2,negcount2,poscount3,negcount3 = 0,0,0,0,0,0  # 
                if word in hp.posdict : 
                    if word in ['好','真','实在'] and words[min(i+1,len(words)-1)] in hp.pos_neg_dict  and words[min(i+1,len(words)-1)] != word:
                        continue
                    else:
                        poscount +=1
                        c = 0
                        for w in words[a:i]: # 扫描情感词前的程度词
                            if w in hp.mostdict:
                                poscount *= 4
                            elif w in hp.verydict:
                                poscount *= 3 
                            elif w in hp.moredict:
                                poscount *= 2 
                            elif w in hp.ishdict:
                                poscount *= 0.5
                            elif w in hp.insufficientlydict:
                                poscount *= -0.3 
                            elif w in hp.overdict:
                                poscount *= -0.5 
                            elif w in hp.inversedict: 
                                c+= 1
                            else:
                                poscount *= 1
                        if tool.is_odd(c) == 'odd': # 扫描情感词前的否定词数
                            poscount *= -1.0
                            poscount2 += poscount
                            poscount = 0
                            poscount3 = poscount + poscount2 + poscount3
                            poscount2 = 0
                        else:
                            poscount3 = poscount + poscount2 + poscount3
                            poscount = 0
                        a = i+1
                elif word in hp.negdict: # 消极情感的分析，与上面一致              
                    if word in ['好','真','实在'] and words[min(i+1,len(words)-1)] in hp.pos_neg_dict and words[min(i+1,len(words)-1)] != word:
                        continue
                    else:
                        negcount += 1
                        d = 0
                        for w in words[a:i]:                         
                            if w in hp.mostdict:
                                negcount *= 4
                            elif w in hp.verydict:
                                negcount *= 3
                            elif w in hp.moredict:
                                negcount *= 2
                            elif w in hp.ishdict:
                                negcount *= 0.5
                            elif w in hp.insufficientlydict:
                                negcount *= -0.3
                            elif w in hp.overdict:
                                negcount *= -0.5
                            elif w in hp.inversedict:
                                d += 1
                            else:
                                negcount *= 1
                    if tool.is_odd(d) == 'odd':
                        negcount *= -1.0
                        negcount2 += negcount
                        negcount = 0
                        negcount3 = negcount + negcount2 + negcount3
                        negcount2 = 0
                    else:
                        negcount3 = negcount + negcount2 + negcount3
                        negcount = 0
                    a = i + 1      
                i += 1
                pos_count = poscount3
                neg_count = negcount3
                count1.append([pos_count,neg_count])           
            if words[-1] in ['!','！']:# 扫描感叹号前的情感词，发现后权值*2
                count1 = [[j*2 for j in c] for c in count1]
    
            for w_im in ['但是','但']:
                if w_im in words : # 扫描但是后面的情感词，发现后权值*5
                    ind = words.index(w_im)
                    count1_head = count1[:ind]
                    count1_tail = count1[ind:]            
                    count1_tail_new = [[j*5 for j in c] for c in count1_tail]
                    count1 = []
                    count1.extend(count1_head)
                    count1.extend(count1_tail_new)
                    break          
            if words[-1] in ['?','？']:# 扫描是否有问好，发现后为负面
                count1 = [[0,2]]
    
            count2.append(count1)
            count1=[]
        return count2
      
        
    def sentiment_score(self,s):
        senti_score_list = self.sentiment_score_list(s)
        if senti_score_list != []:
            negatives=[]
            positives=[]
            for review in senti_score_list:
                score_array =  np.array(review)
                AvgPos = np.sum(score_array[:,0])
                AvgNeg = np.sum(score_array[:,1])        
                negatives.append(AvgNeg)
                positives.append(AvgPos)   
            pos_score = np.mean(positives) 
            neg_score = np.mean(negatives)
            if pos_score >=0 and  neg_score<=0:
                pos_score = pos_score
                neg_score = abs(neg_score)
            elif pos_score >=0 and  neg_score>=0:
                pos_score = pos_score
                neg_score = neg_score    
        else:
            pos_score,neg_score=0,0
        return pos_score,neg_score
       
    def normalization_score(self,sent):
        score1,score0 = self.sentiment_score(sent)
        if score1 > 4 and score0 > 4:
            if score1 >= score0:
                _score1 = 1
                _score0 = score0/score1    
            elif score1 < score0:
                _score0 = 1
                _score1 = score1/score0  
        else :
            if score1 >= 4 :
                _score1 = 1
            elif score1 < 4 :
                _score1 = score1/4
            if score0 >= 4 :
                _score0 = 1
            elif score0 < 4 :
                _score0 = score0/4 
        return _score1,_score0
        



if __name__ =='__main__':
    sa = SentimentAnalysis()
    text = '我妈说明儿不让出去玩'
    print(sa.normalization_score(text))




