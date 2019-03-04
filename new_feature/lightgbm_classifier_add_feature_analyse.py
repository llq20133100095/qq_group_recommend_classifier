# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 15:22:24 2019

@author: leolqli
@function: 
    1.analyse the 400 samples in "analyse data"
    2.the data is "tags.50.filt.removeall.v6.del_sig_key.csv"
    3.set the missing data: "./analyse_data/train_data_no_missing.csv"
    4.random sampling the threshold on test data
"""

import numpy as np
import pandas as pd
from lightgbm_classifier_add_feature import load_pos_dict, load_emb_dict, print_results
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, roc_curve, auc
import time
from svm_classifier import load_pos_dict_one_hot
import matplotlib.pyplot as plt

def sample_data(file_name, sample_number, save_file):
    """
    Function:
        get the sample from the test_file
    """
    dataframe_test = pd.read_csv(file_name, encoding='utf-8')
    dataframe_test = dataframe_test[dataframe_test['pre_label']==1]
    dataframe_test.reset_index(drop=True, inplace=True)
    
    number_list = []
    for i in range(sample_number):
        number_list.append(np.random.randint(0, len(dataframe_test)-1))
    dataframe_sample = dataframe_test.ix[number_list]
    dataframe_sample.to_csv(save_file, index=None, encoding='utf-8-sig')
    return dataframe_test

def sample_threshold_data(file_name, sample_number, save_file, threshold1, threshold2=None):
    """
    Function:
        1.get the sample from the test_file
        2.have the threshold
    """
    dataframe_test = pd.read_csv(file_name, encoding='utf-8')
    dataframe_test = dataframe_test[dataframe_test['predicte'] >= threshold1]
    if(threshold2 != None ):  
        dataframe_test = dataframe_test[dataframe_test['predicte'] < threshold2]

    dataframe_test.reset_index(drop=True, inplace=True)
    
    number_list = []
    for i in range(sample_number):
        while True:
            seed = np.random.randint(0, len(dataframe_test)-1)
            if(seed not in number_list):
                number_list.append(seed)
                break
        
        
    dataframe_sample = dataframe_test.ix[number_list]
    dataframe_sample.to_csv(save_file, index=None, encoding='utf-8-sig')
    return dataframe_test

def get_data_array(file_name, is_training, data_emb, word_emb, word_pos):
    """
    Function:
        1.only has the new feature, no have the embedding
    Input:
        1.file_name: str, the file of "train_data" and "test_data"
        2.is_training: boolean, if training?
        3.data_emb: array, store the data word embeddings
        4.word_emb: dict, store all the word embeddings
        5.word_pos: dict, store all the pos embeddings
    Return:
        1.data_y
        2.np.array(data_emb)
    """
    columns = ['newtitleexttag2count', 'newdescexttag2count', 'classifer', 'entropy', 'entropyv2']
    

    data = pd.read_csv(file_name, encoding='utf-8')
    
    if(is_training):
        #process the null value
        for i in columns:
            data = data[data[i].notnull()]
        data_y = np.array(data[u'remove'])
            
    for index, each_row in data.iterrows():
        each_row_word_emb = 0.0
        word_num = 0.0
        #if the POS no has the null value
        if(not pd.isnull(each_row[u'词性'])):
            parse_list = each_row[u'词性'].split(" ")
            for parse in parse_list:
                word = parse.rsplit("/")[0].encode("utf-8")
                pos = parse.rsplit("/")[-1].encode("utf-8")
                word_num += 1
                if(word in word_emb.keys()):
                    each_row_word_emb += np.concatenate((word_emb[word], word_pos[pos]), axis=0)
                else:
                    each_row_word_emb += np.concatenate((word_emb['UNK'], word_pos[pos]), axis=0)
        else:
            word_num += 1
            each_row_word_emb = np.concatenate((word_emb['UNK'], word_pos['UNK']), axis=0)
        each_row_word_emb /= word_num
        each_row_word_emb = np.concatenate((each_row_word_emb, np.array(each_row[columns])), axis=0)
#        each_row_word_emb = np.array(each_row[columns])
        data_emb.append(each_row_word_emb)
    
    if(is_training):
        return data_y, np.array(data_emb)
    else:
        return np.array(data_emb)
    
def merge_dataframe_one(x, y):
    """
    Function:
        1.if is null, return the other value
    """
    if(pd.isnull(x)):
        return y
    else:
        return x
    
def set_missing_value(file_train, file_train_no_miss):
    """
    Function:
        1.set missing value in train data
    Parameters:
        1.file_train: str, train data
    """
    columns = ['newtag2cout', 'newtitleexttag2count', 'newdescexttag2count', 'classifer', 'entropy', 'entropyv2', 'remove']

    params = {
        "boosting_type": "gbdt",
        "num_leaves": 1000,
        "max_depth": 10,
        "learning_rate": 0.2,
        "n_estimators": 5000,
        "max_bin": 425,
        "subsample_for_bin": 1000,
        "objective": 'regression',
        'metric': 'rmse',
        "min_split_gain": 0,
        "min_child_weight": 0.001,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 1,
        "reg_alpha": 3,
        "reg_lambda": 5,
        "seed": 2019,
        "n_jobs": 5,
        "verbose": 1,
        "silent": False,
    }
    
    data = pd.read_csv(file_train, encoding='utf-8')
    
    for col_num in range(len(columns) - 1):
        #split the train_data and test_data
        columns_id = range(len(columns))
        train_data = data[data[columns[col_num]].notnull()]
        test_data = data[data[columns[col_num]].isnull()]
        
        #get the data and label
        del columns_id[col_num]
        columns_list = []
        for i in columns_id:
            columns_list.append(columns[i])
        train_X = train_data[columns_list]
        train_Y = train_data[columns[col_num]]
        test_X = test_data[columns_list]
        test_Y = test_data[columns[col_num]]
            
        #start the gbm model
        lgb_train = lgb.Dataset(train_X, train_Y)
        lgb_val = lgb.Dataset(train_X, train_Y, reference=lgb_train)
        
        gbm = lgb.train(params, lgb_train, num_boost_round=5000,\
            valid_sets=lgb_val, early_stopping_rounds=100, verbose_eval=50)
        
        #test predicte
        test_pre_value = gbm.predict(test_X, num_iteration=gbm.best_iteration)
        
        #merge the test data in all data
        columns_name = []
        columns_name.append(columns[col_num])
        #reset the index from the test_Y
        test_pre_value = pd.DataFrame(test_pre_value, columns=columns_name)
        test_pre_value.index = test_Y.index
        test_Y = test_pre_value
        data_merge = pd.merge(data, test_Y, how='outer', left_index=True, right_index=True)
        
        x = columns[col_num]+"_x"
        y = columns[col_num]+"_y"
        data[columns[col_num]] = data_merge[[x, y]].apply(lambda fea: merge_dataframe_one(fea[x], fea[y]), axis=1)
    
    data.to_csv(file_train_no_miss, encoding='utf-8-sig', index=None)
    return train_X, train_Y, test_X, test_Y, test_pre_value, data
    
def plot_roc_curve():
    """
    Function:
        1.plot the roc curve
    """
    file_all_fea_embedding = './analyse_data/classifier/tags.50.filt.removeall.v6.del_sig_key.sample400.test_pre_all_feature_word_embedding.csv'
    file_all_fea = './analyse_data/classifier/tags.50.filt.removeall.v6.del_sig_key.sample400.test_pre_All_feature.csv'
    file_no_newtag2cout_embedding = './analyse_data/classifier/tags.50.filt.removeall.v6.del_sig_key.sample400.test_pre_no_newtag2cout_word_embedding.csv'
    file_no_newtag2cout = './analyse_data/classifier/tags.50.filt.removeall.v6.del_sig_key.sample400.test_pre_no_newtag2cout.csv'
    
    data_all_fea_embedding = pd.read_csv(file_all_fea_embedding, encoding="utf-8")
    data_all_fea = pd.read_csv(file_all_fea, encoding="utf-8")
    data_no_newtag2cout_embedding = pd.read_csv(file_no_newtag2cout_embedding, encoding="utf-8")
    data_no_newtag2cout = pd.read_csv(file_no_newtag2cout, encoding="utf-8")

    label = np.array(data_all_fea_embedding['manual_label'])
    test_pre_fin0 = np.array(data_all_fea_embedding['predicte'])
    test_pre_fin1 = np.array(data_all_fea['predicte'])
    test_pre_fin2 = np.array(data_no_newtag2cout_embedding['predicte'])
    test_pre_fin3 = np.array(data_no_newtag2cout['predicte'])
    
    fpr0, tpr0, thresholds = roc_curve(label, test_pre_fin0)
    fpr1, tpr1, thresholds = roc_curve(label, test_pre_fin1)
    fpr2, tpr2, thresholds = roc_curve(label, test_pre_fin2)
    fpr3, tpr3, thresholds = roc_curve(label, test_pre_fin3)
    print("All feature + word embedding: %f" % auc(fpr0, tpr0))
    print("All feature: %f" % auc(fpr1, tpr1))
    print("no newtag2cout+ word embedding: %f" % auc(fpr2, tpr2))
    print("no newtag2cout: %f" % auc(fpr3, tpr3))
    
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.figure(figsize=(8, 8))
    plt.plot(fpr0, tpr0, label='All feature + word embedding')
    plt.plot(fpr1, tpr1, label='All feature')
    plt.plot(fpr2, tpr2, label='no newtag2cout+ word embedding')
    plt.plot(fpr3, tpr3, label='no newtag2cout')
    plt.legend()
    
    plt.savefig('./analyse_data/classifier/roc.jpg')

if __name__ == "__main__":
#    """1.sample the 100 test data"""
#    test_file = './analyse_data/tags.50.filt.removeall.v6.del_sig_key.test_pre.csv'
#    save_file = './analyse_data/tags.50.filt.removeall.v6.del_sig_key.test_pre200_2.csv'
#    dataframe_sample = sample_data(test_file, 200, save_file)
    
    
#    """2.set missing value"""
#    file_train_no_miss = './analyse_data/train_data_no_missing.csv'
#    file_train = './data_process_simple/train_data.csv'
#    train_X, train_Y, test_X, test_Y, test_pre_value, data = set_missing_value(file_train, file_train_no_miss)
    
    '''
    """3.predicte the 400 samples"""
    file_word2vec_save = '../word_embeddings/train_test_word2vec.txt'
#    file_train = './data_process_simple/train_data.csv'
    file_train = './analyse_data/train_data_no_missing.csv'
    file_test = './analyse_data/tags.50.filt.removeall.v6.del_sig_key.sample400.csv'
#    file_test = './data_process_test/tags.50.filt.removeall.v6.del_sig_key.test.csv'
    file_pos = '../data/POS.txt'
    word_emb = {}
    word_pos = {}
    start_time = time.time()
    
    #store the word embeddings in "train_data" and "test_data"
    train_data_emb = []
    test_data_emb = []
    is_training = True
    
#    aa = pd.read_csv(file_train)
#    print aa.info()
    
    #load the word embeddings
    print("Load word embeddings......")
    word_pos = load_pos_dict_one_hot(word_pos, file_pos)
    word_emb = load_emb_dict(word_emb, file_word2vec_save)
    print("Finish the loading: %f s" % (time.time()-start_time))
    
    #train_data word embedding and test_data embedding. Get the label in train_data
    print("Load the data......")
    train_data_y, train_data_x = get_data_array(file_train, True, train_data_emb, word_emb, word_pos)
    test_data_x = get_data_array(file_test, False, test_data_emb, word_emb, word_pos)
    print("Finish the data construction: %f s" % (time.time()-start_time))

    #Cross validation
    print("Start the gbm model......")
    start_time = time.time()
    N = 5
    skf = StratifiedKFold(n_splits=N, random_state=42, shuffle=True)
    params = {
        "boosting_type": "gbdt",
        "num_leaves": 1000,
        "max_depth": 10,
        "learning_rate": 0.05,
        "n_estimators": 5000,
        "max_bin": 425,
        "subsample_for_bin": 1000,
        "objective": 'binary',
        "min_split_gain": 0,
        "min_child_weight": 0.001,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 1,
        "reg_alpha": 3,
        "reg_lambda": 5,
        "seed": 2019,
        "n_jobs": 5,
        "verbose": 1,
        "silent": False,
    }
    
    #train gbm model
    precision_list = []
    recall_list = []
    f1_sco_list = []
    test_pre_list = []
    for k, (train_in, val_in) in enumerate(skf.split(train_data_x, train_data_y)):
        print('train %d flod:' % (k))
#        if(k == 4):
        train_X, val_X, train_Y, val_Y = train_data_x[train_in], train_data_x[val_in], train_data_y[train_in], train_data_y[val_in]
#        train_X, val_X, train_Y, val_Y = train_data_x, train_data_x, train_data_y, train_data_y
            
        #start the gbm model
        lgb_train = lgb.Dataset(train_X, train_Y)
        lgb_val = lgb.Dataset(val_X, val_Y, reference=lgb_train)
        
        gbm = lgb.train(params, lgb_train, num_boost_round=5000,\
            valid_sets=lgb_val, early_stopping_rounds=100, verbose_eval=50)
        
        #train predicte
        train_pre = gbm.predict(train_X, num_iteration=gbm.best_iteration)
        train_pre = np.where(train_pre > 0.6, 1,0)
        precision, recall, f1_sco = print_results(train_Y, train_pre, 'train data')
        
        #val predicte
        val_pre_value = gbm.predict(val_X, num_iteration=gbm.best_iteration)
        val_pre = np.where(val_pre_value > 0.6, 1,0)
        precision, recall, f1_sco = print_results(val_Y, val_pre, 'val data')
        precision_list.append(precision)
        recall_list.append(recall)
        f1_sco_list.append(f1_sco)
        con_mat = confusion_matrix(val_Y, val_pre)
        print(con_mat)
    #    print("Finish the %d gbm model: %f s" % (k, time.time()-start_time))
        
        #test predicte in each fold
        test_pre = gbm.predict(test_data_x, num_iteration=gbm.best_iteration)
        test_pre_list.append(test_pre)
                        

#    #feature important
#    columns = []
#    for i in range(6):
#        columns.append(str(i))
#    fea_import = pd.DataFrame({'column': columns, 'importance': gbm.feature_importance()}).sort_values(by='importance')

    print("-------------Finally-------------")
    print("f1_score: %f " % (np.mean(f1_sco_list)))
    print("precision: %f " % (np.mean(precision_list)))
    print("recall: %f " % (np.mean(recall_list)))
    
    
    """test predicte: sample 400 test data"""
    file_pre = './analyse_data/tags.50.filt.removeall.v6.del_sig_key.sample400.test_pre.csv'
    test_pre_fin = np.mean(test_pre_list, axis=0)
    test_data = pd.read_csv(file_test)
    test_data['predicte'] = pd.DataFrame(test_pre_fin)    
    test_data['pre_label'] = pd.DataFrame(np.where(test_pre_fin > 0.6, 1, 0))
    test_data.to_csv(file_pre, encoding='utf-8-sig', index=None)
    
    label = list(test_data['manual_label'])
    predict = np.where(test_pre_fin > 0.5, 1, 0)
    
    precision = precision_score(label, predict, average=None)
    recall = recall_score(label, predict, average=None)
    f1_sco = f1_score(label, predict, average=None)
    print("-------------Test data-------------")
    print("0 f1_score: %f " % (f1_sco[0]))
    print("0 precision: %f " % (precision[0]))
    print("0 recall: %f " % (recall[0]))
    print("1 f1_score: %f " % (f1_sco[1]))
    print("1 precision: %f " % (precision[1]))
    print("1 recall: %f " % (recall[1]))
    con_mat = confusion_matrix(label, predict)
    print(con_mat)
    '''
    
    """4.sample the 100 in threshold on test_data"""
    test_file = './analyse_data/test3/tags.50.filt.removeall.v6.del_sig_key.test_pre.csv'
    save_file = './analyse_data/test3/tags.50.filt.removeall.v6.del_sig_key.test_pre.100_0_0.1.csv'
    dataframe_sample = sample_threshold_data(test_file, 100, save_file, 0, 0.1)
    
#    """5.print the roc curve"""
#    plot_roc_curve()