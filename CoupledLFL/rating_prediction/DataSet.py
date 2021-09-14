import scipy.sparse as sp
import numpy as np
import csv
# import pandas as pd


# def __init__(self, path, k):
#     '''
#     Constructor
#     '''
#     self.trainMatrix = self.load_rating_file_as_matrix("../preprocessed_data/data0to4607047_index.csv")
#     self.user_review_theta = self.load_review_theta(path + "." + str(k) + ".user.theta")
#     self.item_review_theta = self.load_review_theta(path + "." + str(k) + ".item.theta")
#     self.user_review_fea,self.item_review_fea = self.load_review_fea()
#     self.testRatings = self.load_rating_file_as_matrix(path + ".test.dat")
#     self.num_users, self.num_items = self.trainMatrix.shape

def load_rating_file_as_matrix(filename):
        '''
        Read .rating file and Return dok matrix.
        The first line of .rating file is: num_users\t num_items
        '''
        csv_reader = csv.reader(open(filename, encoding='utf-8'))
        csv_reader1 = csv.reader(open(filename, encoding='utf-8'))
        # Get number of users and items
        num_users, num_items = 0, 0
        for line in csv_reader:
            # arr = line.split(",")
            u, i = int(line[0]), int(line[1])
            num_users = max(num_users, u)#23981
            num_items = max(num_items, i)#98592
        # Construct matrix

        mat = sp.dok_matrix((num_users + 1, num_items + 1), dtype=np.float32)
        for line in csv_reader1:
            user, item, rating = int(line[0]), int(line[1]), float(line[2])
            # print user, item , rating
            if rating > 0:
                mat[user, item] = rating
        return mat

def load_review_feature(filename):
    csv_reader = csv.reader(open(filename, encoding='utf-8'))
    dict = {}
    for line in csv_reader:
        # while line != None and line != "":
        # fea = line.split(',')
        index = int(line[0])
        if index not in dict:
            dict[index] = line[1:]
    return dict


# def init_embeddings_map(fname):
#     with open((fname), encoding="utf8") as glove:
#         return {l[0]: np.asarray(l[1:], dtype="float32") for l in
#                 [line.split() for line in glove]}
#
# def get_embed_and_pad_func(i_seq_len, u_seq_len, pad_value, embedding_map):
#     def embed(row):
#         sentence = row["userReviews"].split()[:u_seq_len]
#         # print(sentence)
#         reviews = list(map(lambda word: embedding_map.get(word)
#         if word in embedding_map else pad_value, sentence))
#         # print(reviews)
#         row["userReviews"] = reviews + \
#                              [pad_value] * (u_seq_len - len(reviews))
#         sentence = row["movieReviews"].split()[:i_seq_len]
#         # print(sentence)
#         reviews = list(map(lambda word: embedding_map.get(word)
#         if word in embedding_map else pad_value, sentence))
#         row["movieReviews"] = reviews + \
#                               [pad_value] * (i_seq_len - len(reviews))
#         return row
#
#     return embed
# def load_review_fea(filename):
#     #max seq_len
#     length=[]
#     length1=[]
#     with open(filename) as f:
#         line = f.readline()
#         while line != None and line != "":
#             line = line[3].split(' ')
#             line1=line[4].split(' ')
#             length.append(len(line))
#             length1.append(len(line1))
#             line = f.readline()
#     u_seq_len=max(length)
#     i_seq_len=max(length1)
#     train = pd.read_csv(filename)
#     # train = raw_data[raw_data["reviewerID"].isin(train_users)]
#     with open(("mymodel.txt"), encoding="utf8") as glove:
#         embedding_map = {l[0]: np.asarray(l[1:], dtype="float32") for l in
#                          [line.split() for line in glove]}
#
#     def embed(row):
#         pad_value = np.array([0.0] * 5)#5是emb_size
#         sentence = row["userReviews"].split()[:u_seq_len]
#         # print(sentence)
#         reviews = list(map(lambda word: embedding_map.get(word)
#         if word in embedding_map else pad_value, sentence))
#         # print(reviews)
#         row["userReviews"] = reviews + \
#                              [pad_value] * (u_seq_len - len(reviews))
#         sentence = row["movieReviews"].split()[:i_seq_len]
#         # print(sentence)
#         reviews = list(map(lambda word: embedding_map.get(word)
#         if word in embedding_map else pad_value, sentence))
#         row["movieReviews"] = reviews + \
#                               [pad_value] * (i_seq_len - len(reviews))
#         return row
#
#     embedding_fn = embed
#     train_embedded = train.apply(embedding_fn, axis=1)
#     user_reviews = np.array(list(train_embedded.loc[:, "userReviews"]))
#     item_reviews = np.array(list(train_embedded.loc[:, "itemReviews"]))
#     return  user_reviews,item_reviews
#
# def load_review_feature(filename,max_seq_len,emb_size):
# #filename格式：1,brother you eat me oo gg time ff practice like
#     c = []
#     with open('mymodel.txt', "r") as f:
#         a = [line.split() for line in f]
#         print(a)
#         for i in range(len(a)):
#             b = [float(f) for f in a[i][1:]]
#             # print(b)
#             # print(b[0])
#             c.append(b)
#         em = {a[i][0]: c[i] for i in range(len(a))}
#         print(em)#单词和相应的词向量
#     csv_reader = csv.reader(open(filename, encoding='utf-8'))
#     dict = {}
#     pad_value = [0.0] * emb_size
#     # length=[]
#     # for l in csv_reader:
#     #     length.append(len(l[1].split(' ')))
#     # max_seq_len=max(length)
#     for line in csv_reader:
#         # while line != None and line != "":
#         # fea = line.split(',')
#         index = int(line[0])#字典键
#         if index not in dict:
#             l = line[1].split(' ')
#             if len(l)<max_seq_len:
#                 for i in range(max_seq_len-len(l)):
#                     l.append('PAD')
#             reviews = list(map(lambda word: em.get(word)
#             if word in em else pad_value, l))
#             dict[index] = reviews#字典值
#     return dict
# def load_max_seq_len():
#     csv_reader = csv.reader(open('', encoding='utf-8'))
#     csv_reader1 = csv.reader(open('', encoding='utf-8'))
#     length = []
#     for l in csv_reader:
#         length.append(len(l[1].split(' ')))
#     for line in csv_reader1:
#         length.append(len(line[1].split(' ')))
#     max_seq_len = max(length)
#     return max_seq_len
