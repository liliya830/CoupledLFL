# import pandas as pd
import numpy as np
import csv
import scipy.sparse as sp

def load_rating_file_as_matrix(filename):
    '''
    Read .rating file and Return dok matrix.
    The first line of .rating file is: num_users\t num_items
    '''
    csv_reader = csv.reader(open(filename, encoding='utf-8'))
    csv_reader1 = csv.reader(open(filename, encoding='utf-8'))
    # Get number of users and items
    num_users, num_items = 0, 0
    #构造矩阵
    for line in csv_reader:
         u, i = int(line[0]), int(line[1])
         num_users = max(num_users, u)#23981
         num_items = max(num_items, i)#98592
    # Construct matrix
    # num_users = 61743
    # num_items = 155459
    mat = sp.dok_matrix((num_users + 1, num_items + 1), dtype=np.float32)
    #填充矩阵内容
    for line in csv_reader1:
        user, item, rating = int(line[0]), int(line[1]), float(line[2])
        # print user, item , rating
        if rating > 0:
            mat[user, item] = 1.0
    return mat

def load_rating_file_as_list(filename):
    '''
    :param filename: 文件名
    :return: 用户、项目ID列表
    '''
    ratingList = []
    csv_reader = csv.reader(open(filename, encoding='utf-8'))
    for line in csv_reader:
        user, item = int(line[0]), int(line[1])
        ratingList.append([user, item])
    return ratingList

def load_negative_file(filename):
    '''
    加载测试负样本
    :param filename: 文件名
    :return: 负样本ID列表
    '''
    negativeList = []
    with open(filename, "r") as f:
        line = f.readline()
        while line != None and line != "":
            line = line.replace('\n', '')
            arr = line.split(",")
            negatives = []
            for x in arr[2:]:
                negatives.append(int(x))
            negativeList.append(negatives)
            line = f.readline()
    return negativeList

def load_review_feature(filename):
    '''
    加载用户评论文本向量
    :param filename: 评论文本文件名
    :return: 一个字典{用户ID:用户评论文本向量}
    '''
    dict = {}
    with open(filename, "r") as f:
        line = f.readline()
        while line != None and line != "":
            fea = line.split(',')
            index = int(fea[0])
            if index not in dict:
                dict[index] = fea[1:]
            line = f.readline()
    return dict



