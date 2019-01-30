# qq_group_recommend_classifier
-------------------------------------
## 一、词向量
    1."./word_embeddings/train_test_word2vec.txt"：获得train_data和test_data每个词的词向量，并存到这里。
    (1)利用的是百度百科的word2vec来进行训练提取的
-------------------------------------
## 二、测试数据
    1.train_data.csv：训练数据4000

    2.test_data.csv：测试数据20462

## 三、输出数据
    1."./predicte_data/test_pre.csv":预测结果
	
    2."./predicte_data/test_pre_label0.csv":预测结果，只保存后面label为0的结果，然后predicate从大到小排序

-------------------------------------
## 四、程序
1.word2vec_process.py:

    基于百度百科的词向量，然后获取train_data和test_data分词之后的词向量

2.lightgbm_classifier.py

    （1）利用交叉验证的方法，把训练数据划分为训练集和验证集

    （2）利用lightgbm构建分类器

-------------------------------------


# qq_group_recommend_classifier_2: 增加了多个标签特征，还是用lightgbm来训练
-------------------------------------
## 一、数据
1. tag_origin.csv：最原始的标签
2. tags.50.filt.removeall.v6.del_sig_key.txt：删除了单个词和去掉了业务关键词的列表
3. 标注部分.xlsx：分为positive和negative数据
4. tag.both+top6k.final.review.raw.del_sig_ns_key.txt：大部分是positive数据
5. tags.50.filt.removeall.v4.alltaginfo.txt：含有特征的数据

## 二、程序
1.data_preprocess.py：数据分成train和test的程序


# qq_group_recommend_classifier_3: 增加了多个标签特征，还是用lightgbm来训练
-------------------------------------
## 一、数据
1. tag_origin.csv：最原始的标签
2. tags.50.filt.removeall.v6.del_sig_key.txt：删除了单个词和去掉了业务关键词的列表
3. 标注部分.xlsx：分为positive和negative数据
4. tag.both+top6k.final.review.raw.del_sig_ns_key.txt：大部分是positive数据
5. tags.50.filt.removeall.v4.alltaginfo.txt：含有特征的数据


文件名 | 数量
---|---
tag_origin.csv | 24462
tags.50.filt.removeall.v6.del_sig_key.txt | 15261
标注部分.xlsx |	Positive: 2977         Negative: 269
tag.both+top6k.final.review.raw.del_sig_ns_key.txt  |	Positive: 4458
tags.50.filt.removeall.v4.alltaginfo.txt |	19758


## 二、预处理方法
1. “tag_origin.csv”和“tags.50.filt.removeall.v6.del_sig_key.txt”做差集，得到大部分都为negative的词语，记为“tags_negative.txt”。

2. “标注部分.xlsx”和“tags_negative.txt”合并，形成训练正例数据“label_positive.txt”和训练负例数据“label_negative.txt”
    
    1）“tags_negative.txt”与“标注部分.xlsx”中的positive做差集，出现共同的数据标为positive, “tags_negative.txt”中剩下的数据则为negative，形成数据“tags_negative_remain.txt”
    
    2）“tags_negative_remain.txt”和“标注部分.xlsx”中的negative做并集，形成训练负例数据“label_negative.txt”
    
    3）“标注部分.xlsx”的positive形成训练正例数据“label_positive.txt”

3.“label_positive.txt和“label_negative.txt”为训练数据，从“tags.50.filt.removeall.v4.alltaginfo.txt“除去训练数据，得到预测数据“test_data.txt”。其中训练数据不在“tags.50.filt.removeall.v4.alltaginfo.txt“中的记为“featureless_train_data.txt”

4，“tags.50.filt.removeall.v6.del_sig_key.txt”和“test_data.txt”做交集，得到新的训练数据“test_data2.txt”


## 三、程序
1.data_preprocess_simple.py
    
    （1）数据分成train和test的程序
    
2.lightgbm_classifier_add_feature.py
    
    （1）利用交叉验证的方法，把训练数据划分为训练集和验证集

    （2）利用lightgbm构建分类器
    
3.lightgbm_classifier_add_feature_sample.py

    （1）利用交叉验证的方法，把训练数据划分为训练集和验证集

    （2）利用lightgbm构建分类器
    
    （3）测试随机抽样的100样本（int test_data）
    
4.svm_classifier.py

    (1)load_pos_dict_one_hot:利用one-hot实现词性标注、
    
    (2)get_data_array:随机组合“词性”、“word embeddings”、“6维特征”
     
## 四、输出文件
1.'predicte_data/test_pre_sample100.csv':随机抽样100样本

2.'predicte_data/test_pre.csv':预测的test_data
    
