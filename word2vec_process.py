# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 11:36:33 2019

@author: leolqli
@Function: 
    1.get the POS in "train_data.csv" and "test_data.csv"
    2.save the word embeddings of POS
@Input File:
    1.'./sgns.baidubaike.bigram-char': chinese word embeddings
    2.'./data/train_data.csv'
    3.'./data/test_data.csv'
@Output File:
    1.'./word_embeddings/train_test_word2vec.txt'
"""

from gensim.models import Word2Vec, KeyedVectors
import pandas as pd
import time

def save_emb_data(file_name, word2vec_save, word2vec_emb):
    """
    according to the "train_data" and "test_data", save their word embeddings
    """
    data = pd.read_csv(file_name, encoding='utf-8')
    for index, each_row in data.iterrows():
        if(not pd.isnull(each_row[u'词性'])):
            parse_list = each_row[u'词性'].split(" ")
            for parse in parse_list:
                word = parse.split("/")[0]
                if(word in word2vec_emb.keys()):
                    emb = str(word2vec_emb[word].tolist())
                    emb = emb.replace('[', '')
                    emb = emb.replace(']', '')
                    emb = emb.replace(',', '')
                    word2vec_save.write(word.encode('utf-8') + "\t" + emb + "\n")
            
if __name__ =='__main__':
    file_word2vec = './word_embeddings/sgns.baidubaike.bigram-char'
    file_train = './data/train_data.csv'
    file_test = './data/test_data.csv'
    file_word2vec_save = './word_embeddings/train_test_word2vec.txt'
    start_time = time.time()
    
    #load the word embedding in "file_word2vec"
    word2vec_emb = {}
    model = KeyedVectors.load_word2vec_format(file_word2vec)
    for word in model.wv.vocab:
        word2vec_emb[word] = model[word]
    
    word2vec_save = open(file_word2vec_save, 'w')  
    #according to the "train_data" and "test_data", save their word embeddings
    save_emb_data(file_train, word2vec_save, word2vec_emb)
    save_emb_data(file_test, word2vec_save, word2vec_emb)
    
    word2vec_save.close()
    
    print("Finish time: %f s" % (time.time() - start_time))