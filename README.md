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
	
    （4）得到输出的两个文件“data_process_test/tag.both+top6k.final.review.raw.del_sig_ns_key.test_pre.csv”和“data_process_test/tags.50.filt.removeall.v6.del_sig_key.test_pre.csv”
    
4.svm_classifier.py

    (1)load_pos_dict_one_hot:利用one-hot实现词性标注、
    
    (2)get_data_array:随机组合“词性”、“word embeddings”、“6维特征”
   
5.lightgbm_classifier_add_feature_merge.py

    (1)融合两个组合，同时进行预测
    (2)这些组合有5和6，3和6
## 四、输出文件
1.'predicte_data/test_pre_sample100.csv':随机抽样100样本

2.'predicte_data/test_pre.csv':预测的test_data
    

# qq群标签分类器分析:lightgbm_classifier_add_feature_analyse.py
## 一、不同维度分析 
1.各个特征对测试集的影响

2.根据最后结果进行排序，给出前面的一些分析

3.观察train data的数据

## 二、各个特征对测试集的影响
1.抽样400个，并人工打上标记。生成文件名为：
./new_feature/analyse_data/tags.50.filt.removeall.v6.del_sig_key.sample400.csv

2.观察6个维度的特征对test data 的影响：针对的是label 1的预测

特征 | 阈值 | precision | recall | f1_score
---|---|---|---|---
All feature	| 0.7 |	0.460751 |	0.870968 |	0.602679
- no newtag2cout |	0.7 |	0.552000 |	0.445161 |	0.492857
- no newtitleexttag2count |	0.7 |	0.464164 |	0.877419 |	0.607143
- no newdescexttag2count |	0.7 |	0.438202 |	0.754839 |	0.554502
- no classifer |	0.7 |	0.422053 |	0.716129 |	0.531100
- no entropy |	0.7 |	0.473310 |	0.858065 |	0.610092
- no entropyv2 |	0.7 |	0.455479 |	0.858065 |	0.595078

3.组合不同的特征进行测试

特征 | 阈值 | precision | recall | f1_score
---|---|---|---|---
newtag2cout + entropy |	0.7 |	0.373272 |	0.522581 |	0.435484
newtag2cout + classifer |	0.7 |	0.414097 |	0.606452 |	0.492147

通过观察6个特征的重要性，得到每个特征的重要程度：

特征 | 重要程度
---|---
newtag2cout |	8362
newtitleexttag2count |	7067
newdescexttag2count |	7325
classifer |	7446
entropy |	6916
entropyv2 |	6418

## 三、数据
1.存在文件夹“./new_feature/analyse_data”

## 四、确定排序的阈值在哪个范围比较可信
1.针对label 1的，每个阈值区间中抽样100

**第一次筛选：**

阈值 | 文件名 | 准确率
---|---|---
大于0.9 |	tags.50.filt.removeall.v6.del_sig_key.test_pre.100_0.9.csv |	95%
0.85-0.9 |	tags.50.filt.removeall.v6.del_sig_key.test_pre.100_0.85_0.9.csv |	79%
0.8-0.85 |	tags.50.filt.removeall.v6.del_sig_key.test_pre.100_0.8_0.85.csv |	74%

**大于0.9：605个**


**第二次筛选：**

阈值 | 文件名 | 准确率
---|---|---
大于0.9 |	tags.50.filt.removeall.v6.del_sig_key.test_pre.100_0.9_0.95.csv |	82%
0.85-0.9 |	tags.50.filt.removeall.v6.del_sig_key.test_pre.100_0.85_0.9.csv |	82%

**这里的准确率都不太高。**

2.针对label 0 的，每个阈值区间中抽样100

**第一次筛选：**

阈值 | 文件名 | 准确率
---|---|---
小于0.1 |	tags.50.filt.removeall.v6.del_sig_key.test_pre.100_0_0.1 |	99%
0.1-0.2 |	tags.50.filt.removeall.v6.del_sig_key.test_pre.100_0.1_0.2.csv |	95%
0.2-0.4 |	tags.50.filt.removeall.v6.del_sig_key.test_pre.100_0.2_0.4.csv |	95%

**0-0.4：9938个**

**第二次筛选：**

阈值 | 文件名 | 准确率
---|---|---
小于0.1 |	tags.50.filt.removeall.v6.del_sig_key.test_pre.100_0_0.1 |	99%
0.1-0.15 |	tags.50.filt.removeall.v6.del_sig_key.test_pre.100_0.1_0.15.csv |	97%
0.15-0.2 |	tags.50.filt.removeall.v6.del_sig_key.test_pre.100_0.15_0.2.csv |	96%
0.2-0.25 |	tags.50.filt.removeall.v6.del_sig_key.test_pre.100_0.2_0.25.csv |	91%

**0-0.2：9747个**

**第三次筛选：**

阈值 | 文件名 | 准确率
---|---|---
小于0.1 |	tags.50.filt.removeall.v6.del_sig_key.test_pre.100_0_0.1 |	97%
0.1-0.15 |	tags.50.filt.removeall.v6.del_sig_key.test_pre.100_0.1_0.15.csv |	97%
0.15-0.2 |	tags.50.filt.removeall.v6.del_sig_key.test_pre.100_0.15_0.2.csv |	93%
0.2-0.25 |	tags.50.filt.removeall.v6.del_sig_key.test_pre.100_0.2_0.25.csv |	95%

**第四次筛选：**

阈值 | 文件名 | 准确率
---|---|---
小于0.1 |	tags.50.filt.removeall.v6.del_sig_key.test_pre.100_0_0.1 |	98%
0.1-0.15 |	tags.50.filt.removeall.v6.del_sig_key.test_pre.100_0.1_0.15.csv |	94%
0.15-0.2 |	tags.50.filt.removeall.v6.del_sig_key.test_pre.100_0.15_0.2.csv |	93%
0.2-0.25 |	tags.50.filt.removeall.v6.del_sig_key.test_pre.100_0.2_0.25.csv |	94%

3.结论：根据预测结果，可以取阈值大于0.9的标记为label 1，而取阈值小于0.4的标记为label 0。这里的数据就有10543。

## 五、观察train data的数据

1.本身positive和negative数据存在近似的字样：

（1）单身、

（2）女汉纸、美女

（3）兼职

2.特征缺失值比较多

（1）train_data数量为7288，缺失值的数量

特征 | 非空的数量
---|---
newtag2cout |	6628
newtitleexttag2count |	6040
newdescexttag2count |	6104
classifer |	5694
entropy |	6040
entropyv2  |	6040

（2）用算法进行预测填充: ./analyse_data/train_data_no_missing.csv

（3）利用得到的填充结果作为train data进行预测

特征 | 阈值 | precision | recall | f1_score
---|---|---|---|---
All feature + word embedding |	0.7 |	0.487500 |	0.754839 |	0.592405
All feature |	0.7 |	0.466912 |	0.819355 |	0.594848
no newtag2cout+ word embedding |	0.7 |	0.606061 |	0.387097 |	0.472441
no newtag2cout |	0.7 |	0.592233 |	0.393548 |	0.472868

制作roc曲线，把数据存到./analyse_data/classifier/

![avatar](./new_feature/analyse_data/classifier/roc.jpg)

特征 | auc
---|---
All feature + word embedding |	0.666136
All feature |	0.662134
no newtag2cout+ word embedding |	0.672615
no newtag2cout |	0.629250

最后用特征**no newtag2cout+ word embedding**来进行预测。

（4）通过填充缺失值，可以提升5%左右的准确率。

## 六、结论
1.6个特征对分类器的重要性基本相同，也就是说明并没有出现比较强的特征可以指导分类器进行预测。

2.怀疑train data标记不是大部分正确。

3.通过在train data上填充缺失值，提升5%左右的准确率。

4.根据预测结果，可以取阈值小于0.2的标记为label 0。这里的数据就有9747。这里面的数据具有较大的置信度。

5.剩下5507可以使用人工标记，或者再利用分类器进行分类。
分类器结束，人工标注规则，进行交接