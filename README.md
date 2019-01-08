# qq_group_recommend_classifier
-------------------------------------
## 数据
1.train_data.csv：训练数据4000
2.test_data.csv：测试数据20462

-------------------------------------
## 程序
1.word2vec_process.py:
    基于百度百科的词向量，然后获取train_data和test_data分词之后的词向量

2.lightgbm_classifier.py
    （1）利用交叉验证的方法，把训练数据划分为训练集和验证集
    （2）利用lightgbm构建分类器