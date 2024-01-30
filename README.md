# Sentiment Analysis: 情感分析

[![Python](https://img.shields.io/badge/python-3.7.6-blue?logo=python&logoColor=FED643)](https://www.python.org/downloads/release/python-376/)
   

<img src="https://github.com/hellonlp/sentiment-analysis/blob/master/imgs/02.png" width="850">

<br/>

## 一、简介
### 1. 文本分类
文本分类是自然语言处理（NLP）最基础核心的任务，或者换句话说，几乎所有的任务都是「分类」任务，或者涉及到「分类」这个概念。情感分析是文本分类中非常主要的一个方向，应用场景非常广泛。
### 2. 情感分析
我们将中文文本情感分析分为三大类型，第一个是应用情感词典和句式结构方法来做的；第二个是使用传统机器学习方法来做的，例如Bayes、SVM等；第三个是应用深度学习的方法来做的，例如LSTM、CNN、LSTM+CNN、BERT+CNN等。  
这三种方法中，第一种不需要人工标注，也不需要训练，第二种和第三种方法都需要人工标注大量的数据，然后做有监督的模型训练。

<br/>

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
基于传统机器学习**bayes**的方法。  

### 3. sentiment_analysis_albert
基于深度学习的方法，使用了语言模型**ALBERT**和下游任务框架**TextCNN**。  

### 4. sentiment_analysis_albert_emoji
基于深度学习的方法，使用了语言模型**ALBERT**和下游任务框架**TextCNN**。    
引入**未知token**（emoji是其中的一种），在微调过程中的同时学习未知token的语义向量，从而达到识别未知token情感语义的目的。  

<br/>
## 参考
[基于词典的文本情感分析（附代码）](https://zhuanlan.zhihu.com/p/142011031)  
[文本分类 [ALBERT+TextCNN] [中文情感分析]（附代码）](https://zhuanlan.zhihu.com/p/149491055)  
[中文情感分析 [emoji 表情符号]](https://zhuanlan.zhihu.com/p/338806367)  
