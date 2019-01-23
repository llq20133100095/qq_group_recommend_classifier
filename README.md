# qq_group_recommend_classifier
-------------------------------------
## 词向量
    1."./word_embeddings/train_test_word2vec.txt"：获得train_data和test_data每个词的词向量，并存到这里
-------------------------------------
## 测试数据
    1.train_data.csv：训练数据4000

    2.test_data.csv：测试数据20462

## 输出数据
    1."./predicte_data/test_pre.csv":预测结果
	
    2."./predicte_data/test_pre_label0.csv":预测结果，只保存后面label为0的结果，然后predicate从大到小排序

-------------------------------------
## 程序
1.word2vec_process.py:

    基于百度百科的词向量，然后获取train_data和test_data分词之后的词向量

2.lightgbm_classifier.py

    （1）利用交叉验证的方法，把训练数据划分为训练集和验证集

    （2）利用lightgbm构建分类器