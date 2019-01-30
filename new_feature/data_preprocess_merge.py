# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 19:30:05 2019

@author: leolqli
@Function:
    merge two prediction
"""
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

def pre_label_check(item):
    """
    Function:
        1.check the label and predicte
        2.True: label == predicte
    """
    label = item[u'remove']
    predicte = item[u'pre_label']
    
    if(label == predicte):
        return True
    else:
        return False
    
def merge_predicte(item):
    """
    Function:
        1.merge two prediction, and get the finally prediction
    """
    label = item[u'remove']
    pre1 = item[u'equal_svm']
    pre2 = item[u'equal_lgbm']
    
    if(pre1 == True or pre2 == True):
        return label
    else:
        return abs(label - 1)
    
def diff_predicte(item):
    pre1 = item[u'equal_svm']
    pre2 = item[u'equal_lgbm']

    if(pre1 and not pre2):
        return True
    elif(not pre1 and pre2):
        return True
    else:
        return False
    
if __name__ == '__main__':
    """label merge: svm and lgbm"""
    file_svm = './data_process_test/svm_val_pre.csv'
    file_lgbm = './data_process_test/lgbm_val_pre.csv'
    
    data_svm = pd.read_csv(file_svm, encoding="utf-8")
    data_lgbm = pd.read_csv(file_lgbm, encoding="utf-8")
    
    data_svm[u'equal_svm'] = data_svm.apply(pre_label_check, axis=1)
    data_svm[u'equal_lgbm'] = data_lgbm.apply(pre_label_check, axis=1)
    
    #merge predicte
    data_svm[u'pre_merge'] = data_svm.apply(merge_predicte, axis=1)
    data_svm[u'different'] = data_svm.apply(diff_predicte, axis=1)
    
    label = list(data_svm[u'remove'])
    predict = list(data_svm[u'pre_merge'])
    
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
    
    data_svm.to_csv('./data_process_test/merge_pre.csv', encoding='utf-8-sig', index=None)