# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 20:17:38 2019

@author: leolqli
@Function:
    1.add the two classifers
"""
from lightgbm_classifier_add_feature import load_pos_dict, load_emb_dict, print_results, get_data_array
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import time
from svm_classifier import load_pos_dict_one_hot
from sklearn import svm

def get_data_array_only_feature(file_name, is_training, data_emb, word_emb, word_pos):
    """
    Function:
        train_data word embedding and test_data embedding. Get the label in train_data
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
    columns = ['newtag2cout', 'newtitleexttag2count', 'newdescexttag2count', 'classifer', 'entropy', 'entropyv2']
    

    data = pd.read_csv(file_name, encoding='utf-8')
    
#    #fill nan value
#    data = data.fillna(data.mean())
    
    if(is_training):
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
        each_row_word_emb = np.array(each_row[columns])
        data_emb.append(each_row_word_emb)
    
    if(is_training):
        return data_y, np.array(data_emb)
    else:
        return np.array(data_emb)

def lgb_classify(train_X, val_X, train_Y, val_Y):
    
    #start the gbm model
    lgb_train = lgb.Dataset(train_X, train_Y)
    lgb_val = lgb.Dataset(val_X, val_Y, reference=lgb_train)
    
    gbm = lgb.train(params, lgb_train, num_boost_round=1000,\
        valid_sets=lgb_val, early_stopping_rounds=100, verbose_eval=50)
    
    #train predicte
    train_pre = gbm.predict(train_X, num_iteration=gbm.best_iteration)

    train_pre = np.where(train_pre > 0.6, 1,0)
    precision, recall, f1_sco = print_results(train_Y, train_pre, 'train data')
    return gbm    
 
    
if __name__ == '__main__':
    file_word2vec_save = '../word_embeddings/train_test_word2vec.txt'
    file_train = './data_process_simple/train_data.csv'
    file_test = './predicte_data/test_pre_sample100.csv'
    file_pos = '../data/POS.txt'
    word_emb = {}
    word_pos = {}
    start_time = time.time()
    
    #store the word embeddings in "train_data" and "test_data"
    train_data_emb = []
    test_data_emb = []
    is_training = True
    

    #load the word embeddings
    print("Load word embeddings......")
    word_pos = load_pos_dict_one_hot(word_pos, file_pos)
    word_emb = load_emb_dict(word_emb, file_word2vec_save)
    print("Finish the loading: %f s" % (time.time()-start_time))
    
    #train_data word embedding and test_data embedding. Get the label in train_data
    print("Load the data......")
    train_data_y, train_data_x = get_data_array(file_train, True, train_data_emb, word_emb, word_pos)
    test_data_x = get_data_array(file_test, False, test_data_emb, word_emb, word_pos)

    train_data_emb = []
    test_data_emb = []
    train_data_y_only_feature, train_data_x_only_feature = get_data_array_only_feature(file_train, True, train_data_emb, word_emb, word_pos)
    test_data_x_only_feature = get_data_array_only_feature(file_test, False, test_data_emb, word_emb, word_pos)
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
        "n_estimators": 1000,
        "max_bin": 425,
        "subsample_for_bin": 20000,
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
        "silent": False
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
        train_X_only_feature, val_X_only_feature, train_Y_only_feature, val_Y_only_feature = train_data_x_only_feature[train_in], train_data_x_only_feature[val_in], train_data_y_only_feature[train_in], train_data_y_only_feature[val_in]

        #train the gbm
        gbm = lgb_classify(train_X, val_X, train_Y, val_Y)
        gbm_only_feature = lgb_classify(train_X_only_feature, val_X_only_feature, train_Y_only_feature, val_Y_only_feature)
#        svc = svm.SVC(probability=True, verbose=True)
#        svc.fit(train_X_only_feature, train_Y_only_feature)
        
        #val predicte
        val_pre_value = gbm.predict(val_X, num_iteration=gbm.best_iteration)

        val_pre_value_only_feature = gbm_only_feature.predict(val_X_only_feature, num_iteration=gbm_only_feature.best_iteration)
#        val_pre_value_only_feature = svc.predict_proba(val_X_only_feature)[:, 1]
        
        #merge two predicte
        val_pre_value = (val_pre_value + val_pre_value_only_feature) / 2

        
        val_pre = np.where(val_pre_value > 0.6, 1,0)
        precision, recall, f1_sco = print_results(val_Y, val_pre, 'val data')
        precision_list.append(precision)
        recall_list.append(recall)
        f1_sco_list.append(f1_sco)
        con_mat = confusion_matrix(val_Y, val_pre)
        print(con_mat)
        print("Finish the %d gbm model: %f s" % (k, time.time()-start_time))
        
        #test predicte in each fold
        test_pre = gbm.predict(test_data_x, num_iteration=gbm.best_iteration)
        test_pre_only_feature = gbm_only_feature.predict(test_data_x_only_feature, num_iteration=gbm_only_feature.best_iteration)
#        test_pre_only_feature = svc.predict_proba(test_data_x_only_feature)[:, 1]
        test_pre = (test_pre + test_pre_only_feature) / 2

        test_pre_list.append(test_pre)
        
#            #save the val data
#            file_val_pre = './data_process_test/lgbm_val_pre_5.csv'
#            val_dataframe = pd.read_csv(file_train, encoding='utf-8')
#            val_dataframe = val_dataframe.ix[val_in]
#            val_dataframe['predicte'] = val_pre_value
#            val_dataframe['pre_label'] = val_pre
#            val_dataframe.to_csv(file_val_pre, encoding='utf-8-sig', index=None)
            

    print("-------------Finally-------------")
    print("f1_score: %f " % (np.mean(f1_sco_list)))
    print("precision: %f " % (np.mean(precision_list)))
    print("recall: %f " % (np.mean(recall_list)))
    
    """test prediction"""
    test_pre_fin = np.mean(test_pre_list, axis=0)
    test_data = pd.read_csv(file_test)
    label = list(test_data[u'manual_label'])
    predict = np.where(test_pre_fin > 0.52, 1, 0)
    
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
            