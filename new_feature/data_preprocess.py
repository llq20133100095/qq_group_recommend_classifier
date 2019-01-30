# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 17:42:00 2019

@author: leolqli
"""
import pandas as pd

def write_in_first_line(file_name, word_after_num):
    """
    insert in the first line
    """
    with open(file_name, 'r+') as f:
        content = f.read()        
        f.seek(0, 0)
        f.write(str(word_after_num) + '\n' + content)
        
def sub_two_file(data1, data2):
#    dif_word = []
#    for word in data1:
#        if(word not in data2):
#            dif_word.append(word)
    
    dif_word = list(set(data1).difference(set(data2)))
    return dif_word, len(dif_word)


def intersection_two_file(data1, data2):
#    int_word = []
#    for word in data1:
#        if(word in data2):
#            int_word.append(word)
            
    int_word = list(set(data1).intersection(set(data2)))
    return int_word, len(int_word)

def union_two_file(data1, data2):
#    union_word = []
#    for word in data1:
#        union_word.append(word)
#        
#    for word in data2:
#        if(word not in union_word):
#            union_word.append(word)
            
    union_word = list(set(data1).union(set(data2)))
    return union_word, len(union_word)

def get_data_in_origin_v6(file_origin, file_del_sig_key, file_dif_origin_v6):
    """
    Function:
        1.Get the data in "tag_origin.csv" and "tags.50.filt.removeall.v6.del_sig_key.txt"
        2.tag_del_sig_key has 15261 in tag_origin, except for 7 words
    
    """
    data_origin = pd.read_csv(file_origin)
    tag_origin = list(data_origin['tag'])
    tag_origin = map(lambda x: str(x).lower(), tag_origin)
    
    line_num = 0
    tag_del_sig_key = []
    with open(file_del_sig_key, "r") as f:
        for line in f.readlines():
            if(line_num == 0):
                line_num += 1
                continue
            each_line = line.strip("\n")
            tag_del_sig_key.append(each_line.lower())
            
    #difference in tag_origin and tag_del_sig_key
    tag_dif_origin_v6, len_dif_origin_v6 = sub_two_file(tag_origin, tag_del_sig_key)
    
    #save the tag_dif_origin_v6:
    save_dif_origin_v6 = open(file_dif_origin_v6, "w")
    
    for word in tag_dif_origin_v6:
        save_dif_origin_v6.write(word + "\n")
    
    save_dif_origin_v6.close()
    write_in_first_line(file_dif_origin_v6, len_dif_origin_v6)
    
    return tag_origin, tag_del_sig_key, tag_dif_origin_v6

def merge_dif_tag_both(file_dif_origin_v6, file_tag_both, file_tag_both_remain):
    """
    Function:
        1.merge the tags_negative.txt and tag.both.txt
        2.“标注部分.xlsx”中的negative和“tag.both.remain.txt”做交集
        3.正例数据“label_positive.txt”和“tags_negative.txt”做交集
    """
    tag_dif_origin_v6 = []
    line_num = 0
    with open(file_dif_origin_v6, "r") as f:
        for line in f.readlines():
            if(line_num == 0):
                line_num += 1
                continue
            tag_dif_origin_v6.append(line.strip("\n").lower())
            
    tag_both = []
    line_num = 0
    with open(file_tag_both, "r") as f:
        for line in f.readlines():
            if(line_num == 0):
                line_num += 1
                continue
            tag_both.append(line.strip("\n").lower())
            
    #intersection
    tag_int_v6_both, len_int_v6_both = intersection_two_file(tag_dif_origin_v6, tag_both)
    
    #delete the intersection, remain is the position in "tag_both"
    save_tag_both_remain = open(file_tag_both_remain, 'w')
    line_num = 0
    for word in tag_both:
        if(word not in tag_int_v6_both):
            save_tag_both_remain.write(word + "\n")
            line_num += 1
    save_tag_both_remain.close()
    
    write_in_first_line(file_tag_both_remain, line_num)
        
    return tag_dif_origin_v6, tag_both, tag_int_v6_both


def union_data(file1, file2, file3):
    """
    Function:
        1.“标注部分.xlsx”中的positive和“tag.both.remain_pos.txt”做并集
        2.“标注部分.xlsx”中的negative和“tags_negative_remain.txt”做并集
    """
    line_num = 0
    tag_data1 = []
    with open(file1, "r") as f:
        for line in f.readlines():
            if(line_num == 0):
                line_num += 1
                continue
            each_line = line.strip("\n")
            tag_data1.append(each_line.lower())
            
    line_num = 0
    tag_data2 = []
    with open(file2, "r") as f:
        for line in f.readlines():
            if(line_num == 0):
                line_num += 1
                continue
            each_line = line.strip("\n")
            tag_data2.append(each_line.lower())
    
    #union
    tag_union, len_union = union_two_file(tag_data1, tag_data2)
    
    #save 
    save_file = open(file3, "w")
    for word in tag_union:
        save_file.write(word + "\n")
    save_file.close()
    write_in_first_line(file3, len(tag_union))
    
    return tag_data1, tag_data2, tag_union
 
def split_train_test(tag_union_position, tag_union_negative, file_alltaginfo):
    
    tag_alltaginfo = {}
    line_num = 0
    with open(file_alltaginfo, "r") as f:
        for line in f.readlines():
            if(line_num == 0):
                line_num += 1
                continue
            each_line = line.strip("\n").split("\t")
            tag_alltaginfo[each_line[0].lower()] = line.strip("\n")
            
    tag_test_data = {}
    
    #merge the tag_union_position and tag_union_negative
    tag_train_data = {}
    for word in tag_union_position:
        tag_train_data[word.lower()] = 0
    for word in tag_union_negative:
        tag_train_data[word.lower()] = 1
        
    #split the test_data
    for word in tag_alltaginfo.keys():
        if(word not in tag_train_data.keys()):
            tag_test_data[word] = tag_alltaginfo[word]
            
    #get in train_data and not in tag_alltaginfo
    tag_featureless_train_data, len_featureless_train_data = sub_two_file(list(tag_train_data.keys()), list(tag_alltaginfo.keys()))
            
    return tag_train_data, tag_alltaginfo, tag_test_data, tag_featureless_train_data
            
def save_train_test_data(tag_train_data, tag_alltaginfo, tag_test_data, \
     tag_featureless_train_data, file_train_data, file_test_data, file_featureless_train_data):
    """
    Function:
        save the train_data and test_data
    """
    save_train_data = open(file_train_data, "w")
    save_test_data = open(file_test_data, "w")
    save_featureless_train_data = open(file_featureless_train_data, "w")
    
    fea_less_pos_num = 0
    fea_less_neg_num = 0
    train_pos_num = 0
    train_neg_num = 0
    for word in tag_train_data:
        if(word not in tag_alltaginfo):
            if(tag_train_data[word] == 1):
                fea_less_neg_num += 1
            else:
                fea_less_pos_num += 1
            save_featureless_train_data.write(word + "\t" + str(tag_train_data[word]) + "\n")
        else:
            if(tag_train_data[word] == 1):
                train_neg_num += 1
            else:
                train_pos_num += 1
            save_train_data.write(word + "\t" + "\t".join(tag_alltaginfo[word].split("\t")[1:]) + "\t" + str(tag_train_data[word]) + "\n")
    
    for word in tag_test_data:
        save_test_data.write(word + "\t" + "\t".join(tag_alltaginfo[word].split("\t")[1:]) + "\n")
        
    save_featureless_train_data.close()
    save_train_data.close()
    save_test_data.close()
    
    write_in_first_line(file_featureless_train_data, "pos:" + str(fea_less_pos_num) + " neg:" + str(fea_less_neg_num))
    write_in_first_line(file_train_data, "pos:" + str(train_pos_num) + " neg:" + str(train_neg_num))
    write_in_first_line(file_test_data, len(tag_test_data.keys()))


if __name__ =='__main__':
    """1."""
    file_origin = './data_new_feature/tag_origin.csv'
    file_del_sig_key = './data_new_feature/tags.50.filt.removeall.v6.del_sig_key.txt'
    file_dif_origin_v6 = './data_process/tags_negative.txt'

    tag_origin, tag_del_sig_key, tag_dif_origin_v6 = get_data_in_origin_v6(file_origin, file_del_sig_key, file_dif_origin_v6)
    
    """2."""
    file_tag_both = './data_new_feature/tag.both+top6k.final.review.raw.del_sig_ns_key.txt'
    file_tag_both_remain = './data_process/tag.both.remain.txt'
    tag_dif_origin_v6, tag_both, tag_int_v6_both = merge_dif_tag_both(file_dif_origin_v6, file_tag_both, file_tag_both_remain)
    
    """3."""
    file_label_positive = './data_new_feature/label_positive.txt'
    file_label_negative = './data_new_feature/label_negative.txt'
    
    #3.1
    file_tag_both_remain_pos = './data_process/tag.both.remain_pos.txt'
    tag_negative, tag_both_remain, tag_inter_negative = merge_dif_tag_both(file_label_negative, file_tag_both_remain, file_tag_both_remain_pos)

    #3.2: set(tag_positive) = 2975
    file_tag_positive = './data_process/label_positive.txt'
    tag_positive, tag_both_remain_pos, tag_union_pos = union_data(file_label_positive, file_tag_both_remain_pos, file_tag_positive)
    
    """4."""
    #4.1
    file_tags_negative_remain = './data_process/tags_negative_remain.txt'
    tag_positive, tag_dif_origin_v6, tag_inter_positive = merge_dif_tag_both(file_tag_positive, file_dif_origin_v6, file_tags_negative_remain)

    #4.2
    file_tag_negative = './data_process/label_negative.txt'
    tag_negative, tags_negative_remain, tag_union_neg = union_data(file_label_negative, file_tags_negative_remain, file_tag_negative)
    
    
    """5."""
    file_alltaginfo = './data_new_feature/tags.50.filt.removeall.v4.alltaginfo.txt'
    file_test_data = './data_process/test_data.txt'
    file_train_data = './data_process/train_data.txt'
    file_featureless_train_data = './data_process/featureless_train_data.txt'
    
    tag_train_data, tag_alltaginfo, tag_test_data, tag_featureless_train_data = split_train_test(tag_union_pos, tag_union_neg, file_alltaginfo)
    #save data
    save_train_test_data(tag_train_data, tag_alltaginfo, tag_test_data, tag_featureless_train_data, file_train_data, file_test_data, file_featureless_train_data)
    

