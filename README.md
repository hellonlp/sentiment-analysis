# 简介
1、本项目是在tensorflow版本1.14.0的基础上做的训练和测试。  
2、本项目为中文的文本情感分析，为多文本分类，一共3个标签：1、0、-1，分别表示正面、中面和负面的情感。  
3、欢迎大家联系我 www.hellonlp.com  
4、albert_small_zh_google对应的百度云下载地址：  
   链接：https://pan.baidu.com/s/1RKzGJTazlZ7y12YRbAWvyA  
   提取码：wuxw  
 
 # 使用方法
 1、准备数据  
 数据格式为：sentiment_analysis_albert/data/sa_test.csv  
 2、参数设置  
 参考脚本 hyperparameters.py，直接修改里面的数值即可。  
 3、训练  
 python train.py  
 4、推理  
 python predict.py  
 
 # 知乎代码解读  
 https://zhuanlan.zhihu.com/p/149491055

   



