# Sentiment Analysis: 情感分析

[![Python](https://img.shields.io/badge/python-3.7.6-blue?logo=python&logoColor=FED643)](https://www.python.org/downloads/release/python-376/)
   

<img src="https://github.com/hellonlp/sentiment-analysis/blob/master/imgs/HELLONLP.png" width="800" height="300">


## 一、简介
### 1. 文本分类
xx  
### 2. 情感分析
xx  



## 二、算法

**4种实现方法**
```
├── sentiment-analysis
    └── sentiment_analysis_dict
    └── sentiment_analysis_bayes
    └── sentiment_analysis_albert
    └── sentiment_analysis_albert_emoji
```

### 1. sentiment_analysis_dict
基于词典的方法。  

### 2. sentiment_analysis_bayes
基于bayes的方法。  

### 3. sentiment_analysis_albert
基于深度学习的方法，使用了语言模型ALBERT和下游任务框架TextCNN。  

### 4. sentiment_analysis_albert_emoji
基于深度学习的方法，使用了语言模型ALBERT和下游任务框架TextCNN。    
引入未知token，在微调过程中的同时学习未知token的语义向量，从而达到识别未知token情感语义的目的。  
