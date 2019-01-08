# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 12:15:07 2019

@author: leolqli
@Function:
    
@Return:
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import time

def load_emb_dict(word_emb):
    """
    Function:
        load the word embeddings
    """
    with open(file_word2vec_save, 'r') as f:
        for each_line in f.readlines():
            each_line = each_line.split("\t")
            word = each_line[0]
            embedding = np.array(each_line[1].strip("\n").split(" "))
            word_emb[word] = np.float64(np.array(embedding))
    
    word_emb['UNK'] = np.random.normal(loc=0.0, scale=0.5, size=300)
    return word_emb

def get_data_array(file_name, is_training, data_emb, word_emb):
    """
    Function:
        train_data word embedding and test_data embedding. Get the label in train_data
    Input:
        1.file_name: str, the file of "train_data" and "test_data"
        2.is_training: boolean, if training?
        3.data_emb: array, store the data word embeddings
        4.word_emb: dict, store all the word embeddings
    Return:
        1.data_y
        2.np.array(data_emb)
    """
    data = pd.read_csv(file_name, encoding='utf-8')
    if(is_training):
        data_y = np.array(data[u'remove'])
    for index, each_row in data.iterrows():
        each_row_word_emb = 0.0
        word_num = 0
        #if the POS no has the null value
        if(not pd.isnull(each_row[u'词性'])):
            parse_list = each_row[u'词性'].split(" ")
            for parse in parse_list:
                word = parse.split("/")[0].encode("utf-8")
                word_num += 1
                if(word in word_emb.keys()):
                    each_row_word_emb += word_emb[word]
                else:
                    each_row_word_emb += word_emb['UNK']
        else:
            word_num += 1
            each_row_word_emb = word_emb['UNK']
        each_row_word_emb /= word_num
        data_emb.append(each_row_word_emb)
    
    if(is_training):
        return data_y, np.array(data_emb)
    else:
        return np.array(data_emb)

if __name__ == '__main__':
    file_word2vec_save = './word_embeddings/train_test_word2vec.txt'
    file_train = './data/train_data.csv'
    file_test = './data/test_data.csv'
    word_emb = {}
    start_time = time.time()
    
    #store the word embeddings in "train_data" and "test_data"
    train_data_emb = []
    test_data_emb = []
    is_training = True
    

    #load the word embeddings
    print("Load word embeddings......")
    word_emb = load_emb_dict(word_emb)
    print("Finish the loading: %f s" % (time.time()-start_time))
    
    #train_data word embedding and test_data embedding. Get the label in train_data
    print("Load the data......")
    train_data_y, train_data_x = get_data_array(file_train, True, train_data_emb, word_emb)
    test_data_x = get_data_array(file_test, True, test_data_emb, word_emb)
    print("Finish the data construction: %f s" % (time.time()-start_time))
    
    