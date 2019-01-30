# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 16:21:20 2019

@author: leolqli
"""
from data_preprocess import union_data, split_train_test, save_train_test_data
import pandas as pd

def get_data(file_label_positive):
    """
    Function:
        get positive data
    """
    tag_positive = []
    line_num = 0
    with open(file_label_positive, "r") as f:
        for line in f.readlines():
            if(line_num == 0):
                line_num += 1
                continue
            tag_positive.append(line.strip("\n").lower())
            
    return tag_positive

def generate_train_test_csv(file_train_data, file_origin, is_training):
    """
    generate the train and test data
    """
    #load the origin data
    data_origin_dict = {}
    data_origin = pd.read_csv(file_origin, encoding="utf-8")
    data_origin = data_origin.fillna(u'')
    for index, data in data_origin.iterrows():
        data_origin_dict[data[u'tag'].encode("utf-8").lower()] = data[u'词性'].encode("utf-8")
    
    #align the train_data and origin data
    line_num = 0
    data_train = []
    with open(file_train_data, "r") as f:
        for line in f.readlines():
            each_row = []
            if(line_num == 0):
                line_num += 1
                continue
            each_line = line.strip("\n").split("\t")
            word = each_line[0].lower()
            
            each_row.append(word)

            each_row.append(data_origin_dict[word])
            
            for feature in each_line[1:]:
                each_row.append(feature)

            data_train.append(each_row)
        
    if(is_training):
        save_data = pd.DataFrame(data_train, columns=['tag', '词性', 'oldtag2count', 'oldexttag2count',	\
          'newtag2cout', 'newtitleexttag2count', 'newdescexttag2count',	'classifer', 'entropy',	'entropyv2', 'remove'])
        save_data.to_csv('./data_process_simple/train_data.csv', encoding='utf_8_sig', index=False)

    else:
        save_data = pd.DataFrame(data_train, columns=['tag', '词性', 'oldtag2count', 'oldexttag2count',	\
          'newtag2cout', 'newtitleexttag2count', 'newdescexttag2count',	'classifer', 'entropy',	'entropyv2'])
        save_data.to_csv('./data_process_simple/test_data.csv', encoding='utf_8_sig', index=False)
    return data_train, data_origin_dict

def intersection_generate_test2(file_test_data, file_del_sig_key):
    """
    Function:
        get the test2_data from the "test_data.csv" and "tags.50.filt.removeall.v6.del_sig_key.txt"
    """
    line_num = 0
    del_sig_key_data = []
    with open(file_del_sig_key, "r") as f:
        for line in f.readlines():
            if(line_num == 0):
                line_num += 1
                continue
            each_line = line.strip("\n").decode("utf-8")
            del_sig_key_data.append(each_line)
            
            
    #load the test_data
    test_dataframe = pd.read_csv(file_test_data, encoding="utf-8")
    test2_list = []
    for index, data in test_dataframe.iterrows():
        if(data[u'tag'] in del_sig_key_data):
            test2_list.append(list(data[0:]))
            
    save_data = pd.DataFrame(test2_list, columns=['tag', '词性', 'oldtag2count', 'oldexttag2count',	\
      'newtag2cout', 'newtitleexttag2count', 'newdescexttag2count',	'classifer', 'entropy',	'entropyv2'])
    save_data.to_csv('./data_process_simple/test_data2.csv', encoding='utf_8_sig', index=False)

    return test2_list
       
def generate_two_test(file_del_sig_ns_key, test_del_sig_ns_key, file_del_sig_key, test_del_sig_key, file_feature, file_pos):
    """
    Function:
        generate the two test data in "file_del_sig_ns_key" and "file_del_sig_key"
    """
    columns = ['tag', '词性', 'oldtag2count', 'oldexttag2count', 'newtag2cout', 'newtitleexttag2count', 'newdescexttag2count', 'classifer', 'entropy', 'entropyv2']
    
    #load the feature in dictionary
    line_num = 0
    data_feature = {}
    with open(file_feature, "r") as f:
        for line in f.readlines():
            if(line_num == 0):
                line_num += 1
                continue
            each_line = line.strip("\n").split("\t")
            data_feature[each_line[0].lower()] = each_line[1:]
            
    #load the pos in tag_origin.csv
    data_pos = {}
    dataframe_pos = pd.read_csv(file_pos, encoding="utf-8")
    dataframe_pos = dataframe_pos.fillna(u'')
    for index, data in dataframe_pos.iterrows():
        data_pos[data[u'tag'].encode("utf-8").lower()] = data[u'词性'].encode("utf-8")
    
    #get the test1 feature
    line_num = 0
    data_del_sig_ns_key = []
    with open(file_del_sig_ns_key, "r") as f:
        for line in f.readlines():
            if(line_num == 0):
                line_num += 1
                continue
            line = line.strip("\n").lower()
            
            if(line in data_feature.keys()):
                line_data = []
                line_data.append(line)
                #have the pos
                if(line in data_pos):
                    line_data.append(data_pos[line])
                    line_data = line_data + data_feature[line]
                else:
                    line_data.append('')
                    line_data = line_data + data_feature[line]
            
                data_del_sig_ns_key.append(line_data)
                
    #get the test2 feature
    line_num = 0
    data_del_sig_key = []
    with open(file_del_sig_key, "r") as f:
        for line in f.readlines():
            if(line_num == 0):
                line_num += 1
                continue
            line = line.strip("\n").lower()
            
            if(line in data_feature.keys()):
                line_data = []
                line_data.append(line)
                #have the pos
                if(line in data_pos):
                    line_data.append(data_pos[line])
                    line_data = line_data + data_feature[line]
                else:
                    line_data.append('')
                    line_data = line_data + data_feature[line]
            
                data_del_sig_key.append(line_data)
    
    dataframe_del_sig_ns_key = pd.DataFrame(data_del_sig_ns_key, columns=columns)        
    dataframe_del_sig_ns_key = dataframe_del_sig_ns_key.replace('NULL', '')
    dataframe_del_sig_ns_key.to_csv(test_del_sig_ns_key, index=None, encoding='utf-8-sig')
    
    dataframe_del_sig_key = pd.DataFrame(data_del_sig_key, columns=columns)        
    dataframe_del_sig_ns_key = dataframe_del_sig_key.replace('NULL', '')
    dataframe_del_sig_key.to_csv(test_del_sig_key, index=None, encoding='utf-8-sig')    
    
    
    return data_del_sig_ns_key, data_del_sig_key, data_pos
    
    
if __name__ == '__main__':
#    """2"""
#    file_label_positive = './data_new_feature/label_positive.txt'
#    file_label_negative = './data_new_feature/label_negative.txt'
#    
#    #2.2
#    file_tags_negative_remain = './data_process_simple/tags_negative_remain.txt'
#    file_tag_negative = './data_process_simple/label_negative.txt'
#    tag_negative, tags_negative_remain, tag_union_neg = union_data(file_label_negative, file_tags_negative_remain, file_tag_negative)
#
#    #2.3
#    tag_positive = get_data(file_label_positive)
#    
#    """3."""
#    file_alltaginfo = './data_new_feature/tags.50.filt.removeall.v4.alltaginfo.txt'
#    file_test_data = './data_process_simple/test_data.txt'
#    file_train_data = './data_process_simple/train_data.txt'
#    file_featureless_train_data = './data_process_simple/featureless_train_data.txt'
#    
#    tag_train_data, tag_alltaginfo, tag_test_data, tag_featureless_train_data = split_train_test(tag_positive, tag_union_neg, file_alltaginfo)
#    #save data
#    save_train_test_data(tag_train_data, tag_alltaginfo, tag_test_data, tag_featureless_train_data, file_train_data, file_test_data, file_featureless_train_data)

#    """get the pos in train_data and test_data"""
#    file_origin = './data_process_simple/origin.csv'
#    file_test_data = './data_process_simple/test_data.txt'
#    file_train_data = './data_process_simple/train_data.txt'
#    
#    data_train, data_origin_dict = generate_train_test_csv(file_train_data, file_origin, True)
#    data_test, _ = generate_train_test_csv(file_test_data, file_origin, False)
    
    
#    """4.get the test_data2"""
#    file_test_data = './data_process_simple/test_data.csv'
#    file_del_sig_key = './data_new_feature/tags.50.filt.removeall.v6.del_sig_key.txt'
#    
#    test2_list = intersection_generate_test2(file_test_data, file_del_sig_key)
    
    """5.generate the two test_data"""
    file_del_sig_ns_key = './data_new_feature/tag.both+top6k.final.review.raw.del_sig_ns_key.txt'
    test_del_sig_ns_key = './data_process_test/tag.both+top6k.final.review.raw.del_sig_ns_key.test.csv'
    file_del_sig_key = './data_new_feature/tags.50.filt.removeall.v6.del_sig_key.txt'
    test_del_sig_key = './data_process_test/tags.50.filt.removeall.v6.del_sig_key.test.csv'
    file_feature = './data_new_feature/tags.50.filt.removeall.v4.alltaginfo.txt'
    file_pos = './data_new_feature/tag_origin.csv'
    
    data_del_sig_ns_key, data_del_sig_key, data_pos = generate_two_test(file_del_sig_ns_key, test_del_sig_ns_key, file_del_sig_key, test_del_sig_key, file_feature, file_pos)
    
    
