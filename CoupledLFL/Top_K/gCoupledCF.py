# coding=UTF-8
import gc
import time
from time import time
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.initializers import RandomNormal
from keras.layers import Dense, Activation, Flatten, Lambda, Reshape, MaxPooling2D, AveragePooling2D
from keras.layers import Embedding, Input, merge, Conv2D,Multiply,Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
# from keras.utils import plot_model

# from LoadData import load_review_fea
from .LoadData import load_rating_file_as_matrix
from .LoadData import load_review_feature
from .LoadData import load_rating_file_as_list
from .LoadData import load_negative_file
from .evaluate_DeepCF import evaluate_model
# from create_vocab import create_vocab
# from LoadData import load_max_seq_len

def get_train_instances(ratings,user_review_fea,item_review_fea):

    user_fea, item_fea,labels = [], [], []
    num_users, num_items = ratings.shape
    # num_users=num_users-1
    num_items=num_items-1
    num_negatives = 4
    # print("num_users",num_users)
    # print("num_items",num_items)

    for (u, i) in ratings.keys():
        # positive instance
        # user_vec_input.append(users_vec_mat[u])
        # user_fea_input.append(users_attr_mat[u])
        # user_id_input.append([u])
        user_fea.append(user_review_fea[u])
        # item_id_input.append([i])
        item_fea.append(item_review_fea[i])
        # item_fea_input.append(items_genres_mat[i])
        labels.append([1])

        # negative instances
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            while (u, j) in ratings:
                j = np.random.randint(num_items)

            # user_vec_input.append(users_vec_mat[u])
            # user_fea_input.append(users_attr_mat[u])
            # user_id_input.append([u])
            user_fea.append(user_review_fea[u])
            # item_id_input.append([j])
            # print("j",j)
            item_fea.append(item_review_fea[j])
            # item_fea_input.append(items_genres_mat[j])
            labels.append([0])
    # array_user_vec_input = np.array(user_vec_input)
    array_user_fea_input = np.array(user_fea)
    # array_user_id_input = np.array(user_id_input)
    # array_item_id_input = np.array(item_id_input)
    array_item_fea_input = np.array(item_fea)
    array_labels = np.array(labels)

    del  user_id_input, item_id_input, labels
    gc.collect()

    return array_user_fea_input, array_item_fea_input, array_labels
def get_model(num_users, num_items,max_seq_len):
    ########################   attr side   ##################################

    # Input
    user_fea_input = Input(shape=(max_seq_len,), dtype='float32', name='user_fea_input')
    # word_emb = Embedding(vocab_size, emb_size, name='word_emb')
    # user_attr_embedding=Flatten(name='flatten_user_fea')(word_emb(user_fea_input))#(?,200,100)
    user_attr_embedding = Dense(8,name='dense_user_fea')(user_fea_input)#(?,200,8)
    user_attr_embedding = BatchNormalization()(user_attr_embedding)
    user_attr_embedding = Activation(activation='relu')(user_attr_embedding)
    user_attr_embedding = Reshape((1,8),name='reshape_user_fea')(user_attr_embedding)

    item_fea_input = Input(shape=(max_seq_len,), dtype='float32', name='item_fea_input')
    # item_attr_embedding=Flatten(name='flatten_item_fea')(word_emb(item_fea_input))
    item_attr_embedding = Dense(8,name='dense_item_fea')(item_fea_input)
    item_attr_embedding = BatchNormalization()(item_attr_embedding)
    item_attr_embedding = Activation(activation='relu')(item_attr_embedding)
    item_attr_embedding = Reshape((8, 1),name='reshape_item_fea')(item_attr_embedding)

    # merge_attr_embedding=Multiply()(user_fea_input,item_fea_input)
    # merge_attr_embedding = merge([user_fea_input, item_fea_input], mode='mul')  # element-wise multiply

    merge_attr_embedding = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[1, 2]),name='lambda_merge')(
        [user_attr_embedding, item_attr_embedding])

    merge_attr_embedding_global = Flatten(name='flatten_merge_attr_global')(merge_attr_embedding)

    merge_attr_embedding = Reshape((8, 8, 1),name='reshape_merge_attr')(merge_attr_embedding)
#用一层卷积还是多层卷积需要尝试
    merge_attr_embedding = Conv2D(8, (3, 3),name='conv')(merge_attr_embedding)
    merge_attr_embedding = BatchNormalization(axis=3,name='normalization')(merge_attr_embedding)#归一化
    #这里加不加最大池化，dense层等需要尝试
    merge_attr_embedding = Activation('relu',name='activation_merge_attr')(merge_attr_embedding)#local_vector
    # merge_attr_embedding = AveragePooling2D((2, 2))(merge_attr_embedding)
    # merge_attr_embedding = Dropout(0.35)(merge_attr_embedding)

    # merge_attr_embedding = Conv2D(32, (3, 3))(merge_attr_embedding)
    # merge_attr_embedding = BatchNormalization(axis=3)(merge_attr_embedding)
    # merge_attr_embedding = Activation('relu')(merge_attr_embedding)
    # merge_attr_embedding = MaxPooling2D((2, 2))(merge_attr_embedding)

    # merge_attr_embedding = Conv2D(8, (3, 3))(merge_attr_embedding)
    # merge_attr_embedding = BatchNormalization(axis=3)(merge_attr_embedding)
    # merge_attr_embedding = Activation('relu')(merge_attr_embedding)

    merge_attr_embedding = Flatten(name='flatten_merge_attr')(merge_attr_embedding)
    merge_attr_embedding = merge([merge_attr_embedding, merge_attr_embedding_global], mode='concat',name='merge')

    attr_1 = Dense(16,name='dense_attr_1')(merge_attr_embedding)#16隐藏节点的个数需要调
    #    attr_1=BatchNormalization()(attr_1)#是否加归一化需要尝试
    attr_1 = Activation('relu',name='activation_attr1')(attr_1)
    #    attr_1=Dropout(0.2)(attr_1)

    # attr_2 = Dense(16)(attr_1)
    # attr_2 = Activation('relu')(attr_2)
    #    id_2=BatchNormalization()(id_2)
    #    id_2=Dropout(0.2)(id_2)

    ########################   id side   ##################################

    user_id_input = Input(shape=(1,), dtype='float32', name='user_id_input')
    user_id_Embedding = Embedding(input_dim=num_users, output_dim=32, name='user_id_Embedding',
                                  embeddings_initializer=RandomNormal(
                                      mean=0.0, stddev=0.01, seed=None),
                                  W_regularizer=l2(0), input_length=1)
    user_id_Embedding = Flatten(name='flatten_user_id')(user_id_Embedding(user_id_input))

    item_id_input = Input(shape=(1,), dtype='float32', name='item_id_input')
    item_id_Embedding = Embedding(input_dim=num_items, output_dim=32, name='item_id_Embedding',
                                  embeddings_initializer=RandomNormal(
                                      mean=0.0, stddev=0.01, seed=None),
                                  W_regularizer=l2(0), input_length=1)
    item_id_Embedding = Flatten(name='flatten_item_id')(item_id_Embedding(item_id_input))

    # id merge embedding
    merge_id_embedding = merge([user_id_Embedding, item_id_Embedding], mode='mul',name='merge_id')
    # id_1 = Dense(64)(merge_id_embedding)
    # id_1 = Activation('relu')(id_1)
    #MLP部分，几层还需要尝试
    id_2 = Dense(32,name='dense_id_2')(merge_id_embedding)
    id_2 = Activation('relu',name='activation_id_2')(id_2)

    # merge attr_id embedding
    merge_attr_id_embedding = merge([attr_1, id_2], mode='concat',name='merge_attr_id')
    dense_1 = Dense(64,name='dense_merge_attr_id')(merge_attr_id_embedding)
    dense_1 = Activation('relu',name='activation')(dense_1)
    # dense_1=BatchNormalization()(dense_1)
    #    dense_1=Dropout(0.2)(dense_1)

    # dense_2=Dense(16)(dense_1)
    # dense_2=Activation('relu')(dense_2)
    #    dense_2=BatchNormalization()(dense_2)
    #    dense_2=Dropout(0.2)(dense_2)

    # dense_3=Dense(8)(dense_2)
    # dense_3=Activation('relu')(dense_3)
    #    dense_3=BatchNormalization()(dense_3)
    #    dense_3=Dropout(0.2)(dense_3)

    topLayer = Dense(1, activation='sigmoid', init='lecun_uniform',
                     name='topLayer')(dense_1)

    # Final prediction layer
    model = Model(input=[user_fea_input, item_fea_input, user_id_input, item_id_input],
                  output=topLayer)
    # if emb_path:#None
    #     from w2vEmbReader import W2VEmbReader as EmbReader
    #     emb_reader = EmbReader(emb_path, emb_dim=emb_size)
    #     print('Initializing word embedding matrix')
    #     model.get_layer('word_emb').set_weights(emb_reader.get_emb_matrix_given_vocab(vocab, model.get_layer('word_emb').get_weights()))
    #

    return model


def get_model1(num_users, num_items,max_seq_len):
    ########################   attr side   ##################################

    # Input
    user_fea_input = Input(shape=(max_seq_len,), dtype='float32', name='user_fea_input')
    # word_emb = Embedding(vocab_size, emb_size, name='word_emb')
    # user_attr_embedding=Flatten(name='flatten_user_fea')(word_emb(user_fea_input))#(?,200,100)
    user_attr_embedding = Dense(16, activation='relu',name='dense_user_fea')(user_fea_input)#(?,200,8)
    user_attr_embedding = Reshape((1,16),name='reshape_user_fea')(user_attr_embedding)

    item_fea_input = Input(shape=(max_seq_len,), dtype='float32', name='item_fea_input')
    # item_attr_embedding=Flatten(name='flatten_item_fea')(word_emb(item_fea_input))
    item_attr_embedding = Dense(16, activation='relu',name='dense_item_fea')(item_fea_input)
    item_attr_embedding = Reshape((16, 1),name='reshape_item_fea')(item_attr_embedding)

    # merge_attr_embedding=Multiply()(user_fea_input,item_fea_input)
    # merge_attr_embedding = merge([user_fea_input, item_fea_input], mode='mul')  # element-wise multiply

    merge_attr_embedding = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[1, 2]),name='lambda_merge')(
        [user_attr_embedding, item_attr_embedding])

    merge_attr_embedding_global = Flatten(name='flatten_merge_attr_global')(merge_attr_embedding)

    merge_attr_embedding = Reshape((16, 16, 1),name='reshape_merge_attr')(merge_attr_embedding)
#用一层卷积还是多层卷积需要尝试
    merge_attr_embedding = Conv2D(16, (3, 3),name='conv')(merge_attr_embedding)
    merge_attr_embedding = BatchNormalization(axis=3,name='normalization')(merge_attr_embedding)#归一化
    #这里加不加最大池化，dense层等需要尝试
    merge_attr_embedding = Activation('relu',name='activation_merge_attr')(merge_attr_embedding)#local_vector
    # merge_attr_embedding = AveragePooling2D((2, 2))(merge_attr_embedding)
    # merge_attr_embedding = Dropout(0.35)(merge_attr_embedding)

    # merge_attr_embedding = Conv2D(32, (3, 3))(merge_attr_embedding)
    # merge_attr_embedding = BatchNormalization(axis=3)(merge_attr_embedding)
    # merge_attr_embedding = Activation('relu')(merge_attr_embedding)
    # merge_attr_embedding = MaxPooling2D((2, 2))(merge_attr_embedding)

    # merge_attr_embedding = Conv2D(8, (3, 3))(merge_attr_embedding)
    # merge_attr_embedding = BatchNormalization(axis=3)(merge_attr_embedding)
    # merge_attr_embedding = Activation('relu')(merge_attr_embedding)

    merge_attr_embedding = Flatten(name='flatten_merge_attr')(merge_attr_embedding)
    merge_attr_embedding = merge([merge_attr_embedding, merge_attr_embedding_global], mode='concat',name='merge')

    attr_1 = Dense(32,name='dense_attr_1')(merge_attr_embedding)#16隐藏节点的个数需要调
    #    attr_1=BatchNormalization()(attr_1)#是否加归一化需要尝试
    attr_1 = Activation('relu',name='activation_attr1')(attr_1)
    #    attr_1=Dropout(0.2)(attr_1)

    # attr_2 = Dense(16)(attr_1)
    # attr_2 = Activation('relu')(attr_2)
    #    id_2=BatchNormalization()(id_2)
    #    id_2=Dropout(0.2)(id_2)

    ########################   id side   ##################################

    user_id_input = Input(shape=(1,), dtype='float32', name='user_id_input')
    user_id_Embedding = Embedding(input_dim=num_users, output_dim=32, name='user_id_Embedding',
                                  embeddings_initializer=RandomNormal(
                                      mean=0.0, stddev=0.01, seed=None),
                                  W_regularizer=l2(0), input_length=1)
    user_id_Embedding = Flatten(name='flatten_user_id')(user_id_Embedding(user_id_input))

    item_id_input = Input(shape=(1,), dtype='float32', name='item_id_input')
    item_id_Embedding = Embedding(input_dim=num_items, output_dim=32, name='item_id_Embedding',
                                  embeddings_initializer=RandomNormal(
                                      mean=0.0, stddev=0.01, seed=None),
                                  W_regularizer=l2(0), input_length=1)
    item_id_Embedding = Flatten(name='flatten_item_id')(item_id_Embedding(item_id_input))

    # id merge embedding
    merge_id_embedding = merge([user_id_Embedding, item_id_Embedding], mode='mul',name='merge_id')
    # id_1 = Dense(64)(merge_id_embedding)
    # id_1 = Activation('relu')(id_1)
    #MLP部分，几层还需要尝试
    id_2 = Dense(32,name='dense_id_2')(merge_id_embedding)
    id_2 = Activation('relu',name='activation_id_2')(id_2)

    # merge attr_id embedding
    merge_attr_id_embedding = merge([attr_1, id_2], mode='concat',name='merge_attr_id')
    dense_1 = Dense(64,name='dense_merge_attr_id')(merge_attr_id_embedding)
    dense_1 = Activation('relu',name='activation')(dense_1)
    # dense_1=BatchNormalization()(dense_1)
    #    dense_1=Dropout(0.2)(dense_1)

    # dense_2=Dense(16)(dense_1)
    # dense_2=Activation('relu')(dense_2)
    #    dense_2=BatchNormalization()(dense_2)
    #    dense_2=Dropout(0.2)(dense_2)

    # dense_3=Dense(8)(dense_2)
    # dense_3=Activation('relu')(dense_3)
    #    dense_3=BatchNormalization()(dense_3)
    #    dense_3=Dropout(0.2)(dense_3)

    topLayer = Dense(1, activation='sigmoid', init='lecun_uniform',
                     name='topLayer')(dense_1)

    # Final prediction layer
    model = Model(input=[user_fea_input, item_fea_input, user_id_input, item_id_input],
                  output=topLayer)
    # if emb_path:#None
    #     from w2vEmbReader import W2VEmbReader as EmbReader
    #     emb_reader = EmbReader(emb_path, emb_dim=emb_size)
    #     print('Initializing word embedding matrix')
    #     model.get_layer('word_emb').set_weights(emb_reader.get_emb_matrix_given_vocab(vocab, model.get_layer('word_emb').get_weights()))


    return model

def get_model2(num_users, num_items,max_seq_len):
    ########################   attr side   ##################################

    # Input
    user_fea_input = Input(shape=(max_seq_len,), dtype='float32', name='user_fea_input')
    # word_emb = Embedding(vocab_size, emb_size, name='word_emb')
    # user_attr_embedding=Flatten(name='flatten_user_fea')(word_emb(user_fea_input))#(?,200,100)
    user_attr_embedding = Dense(16, name='dense_user_fea')(user_fea_input)#(?,200,8)
    user_attr_embedding = BatchNormalization()(user_attr_embedding)
    user_attr_embedding = Activation(activation='relu')(user_attr_embedding)
    user_attr_embedding = Reshape((1,16),name='reshape_user_fea')(user_attr_embedding)

    item_fea_input = Input(shape=(max_seq_len,), dtype='float32', name='item_fea_input')
    # item_attr_embedding=Flatten(name='flatten_item_fea')(word_emb(item_fea_input))
    item_attr_embedding = Dense(16,name='dense_item_fea')(item_fea_input)
    item_attr_embedding = BatchNormalization()(item_attr_embedding)
    item_attr_embedding = Activation(activation='relu')(item_attr_embedding)
    item_attr_embedding = Reshape((16, 1),name='reshape_item_fea')(item_attr_embedding)

    # merge_attr_embedding=Multiply()(user_fea_input,item_fea_input)
    # merge_attr_embedding = merge([user_fea_input, item_fea_input], mode='mul')  # element-wise multiply

    merge_attr_embedding = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[1, 2]),name='lambda_merge')(
        [user_attr_embedding, item_attr_embedding])

    merge_attr_embedding_global = Flatten(name='flatten_merge_attr_global')(merge_attr_embedding)

    merge_attr_embedding = Reshape((16, 16, 1),name='reshape_merge_attr')(merge_attr_embedding)
#用一层卷积还是多层卷积需要尝试
    merge_attr_embedding = Conv2D(16, (3, 3),name='conv')(merge_attr_embedding)
    merge_attr_embedding = BatchNormalization(axis=3,name='normalization')(merge_attr_embedding)#归一化
    #这里加不加最大池化，dense层等需要尝试
    merge_attr_embedding = Activation('relu',name='activation_merge_attr')(merge_attr_embedding)#local_vector
    # merge_attr_embedding = AveragePooling2D((2, 2))(merge_attr_embedding)
    # merge_attr_embedding = Dropout(0.35)(merge_attr_embedding)

    # merge_attr_embedding = Conv2D(32, (3, 3))(merge_attr_embedding)
    # merge_attr_embedding = BatchNormalization(axis=3)(merge_attr_embedding)
    # merge_attr_embedding = Activation('relu')(merge_attr_embedding)
    # merge_attr_embedding = MaxPooling2D((2, 2))(merge_attr_embedding)

    # merge_attr_embedding = Conv2D(8, (3, 3))(merge_attr_embedding)
    # merge_attr_embedding = BatchNormalization(axis=3)(merge_attr_embedding)
    # merge_attr_embedding = Activation('relu')(merge_attr_embedding)

    merge_attr_embedding = Flatten(name='flatten_merge_attr')(merge_attr_embedding)
    merge_attr_embedding = merge([merge_attr_embedding, merge_attr_embedding_global], mode='concat',name='merge')

    attr_1 = Dense(32,name='dense_attr_1')(merge_attr_embedding)#16隐藏节点的个数需要调
    attr_1=BatchNormalization()(attr_1)#是否加归一化需要尝试
    attr_1 = Activation('relu',name='activation_attr1')(attr_1)
    #    attr_1=Dropout(0.2)(attr_1)

    # attr_2 = Dense(16)(attr_1)
    # attr_2 = Activation('relu')(attr_2)
    #    id_2=BatchNormalization()(id_2)
    #    id_2=Dropout(0.2)(id_2)

    ########################   id side   ##################################

    user_id_input = Input(shape=(1,), dtype='float32', name='user_id_input')
    user_id_Embedding = Embedding(input_dim=num_users, output_dim=32, name='user_id_Embedding',
                                  embeddings_initializer=RandomNormal(
                                      mean=0.0, stddev=0.01, seed=None),
                                  W_regularizer=l2(0), input_length=1)
    user_id_Embedding = Flatten(name='flatten_user_id')(user_id_Embedding(user_id_input))

    item_id_input = Input(shape=(1,), dtype='float32', name='item_id_input')
    item_id_Embedding = Embedding(input_dim=num_items, output_dim=32, name='item_id_Embedding',
                                  embeddings_initializer=RandomNormal(
                                      mean=0.0, stddev=0.01, seed=None),
                                  W_regularizer=l2(0), input_length=1)
    item_id_Embedding = Flatten(name='flatten_item_id')(item_id_Embedding(item_id_input))

    # id merge embedding
    merge_id_embedding = merge([user_id_Embedding, item_id_Embedding], mode='mul',name='merge_id')
    # id_1 = Dense(64)(merge_id_embedding)
    # id_1 = Activation('relu')(id_1)
    #MLP部分，几层还需要尝试
    id_2 = Dense(32,name='dense_id_2')(merge_id_embedding)
    id_2 = BatchNormalization()(id_2)
    id_2 = Activation('relu',name='activation_id_2')(id_2)

    # merge attr_id embedding
    merge_attr_id_embedding = merge([attr_1, id_2], mode='concat',name='merge_attr_id')
    dense_1 = Dense(64,name='dense_merge_attr_id')(merge_attr_id_embedding)
    dense_1 = BatchNormalization()(dense_1)
    dense_1 = Activation('relu',name='activation')(dense_1)
    # dense_1=BatchNormalization()(dense_1)
    #    dense_1=Dropout(0.2)(dense_1)

    # dense_2=Dense(16)(dense_1)
    # dense_2=Activation('relu')(dense_2)
    #    dense_2=BatchNormalization()(dense_2)
    #    dense_2=Dropout(0.2)(dense_2)

    # dense_3=Dense(8)(dense_2)
    # dense_3=Activation('relu')(dense_3)
    #    dense_3=BatchNormalization()(dense_3)
    #    dense_3=Dropout(0.2)(dense_3)

    topLayer = Dense(1, activation='sigmoid', init='lecun_uniform',
                     name='topLayer')(dense_1)

    # Final prediction layer
    model = Model(input=[user_fea_input, item_fea_input, user_id_input, item_id_input],
                  output=topLayer)
    # if emb_path:#None
    #     from w2vEmbReader import W2VEmbReader as EmbReader
    #     emb_reader = EmbReader(emb_path, emb_dim=emb_size)
    #     print('Initializing word embedding matrix')
    #     model.get_layer('word_emb').set_weights(emb_reader.get_emb_matrix_given_vocab(vocab, model.get_layer('word_emb').get_weights()))
    return model


def get_model3(num_users, num_items,max_seq_len):
    ########################   attr side   ##################################

    # Input
    user_fea_input = Input(shape=(max_seq_len,), dtype='float32', name='user_fea_input')
    # word_emb = Embedding(vocab_size, emb_size, name='word_emb')
    # user_attr_embedding=Flatten(name='flatten_user_fea')(word_emb(user_fea_input))#(?,200,100)
    user_attr_embedding = Dense(16, name='dense_user_fea')(user_fea_input)#(?,200,8)
    user_attr_embedding = BatchNormalization()(user_attr_embedding)
    user_attr_embedding = Activation(activation='relu')(user_attr_embedding)
    user_attr_embedding = Reshape((1,16),name='reshape_user_fea')(user_attr_embedding)

    item_fea_input = Input(shape=(max_seq_len,), dtype='float32', name='item_fea_input')
    # item_attr_embedding=Flatten(name='flatten_item_fea')(word_emb(item_fea_input))
    item_attr_embedding = Dense(16,name='dense_item_fea')(item_fea_input)
    item_attr_embedding = BatchNormalization()(item_attr_embedding)
    item_attr_embedding = Activation(activation='relu')(item_attr_embedding)
    item_attr_embedding = Reshape((16, 1),name='reshape_item_fea')(item_attr_embedding)

    # merge_attr_embedding=Multiply()(user_fea_input,item_fea_input)
    # merge_attr_embedding = merge([user_fea_input, item_fea_input], mode='mul')  # element-wise multiply

    merge_attr_embedding = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[1, 2]),name='lambda_merge')(
        [user_attr_embedding, item_attr_embedding])

    merge_attr_embedding_global = Flatten(name='flatten_merge_attr_global')(merge_attr_embedding)

    merge_attr_embedding = Reshape((16, 16, 1),name='reshape_merge_attr')(merge_attr_embedding)
#用一层卷积还是多层卷积需要尝试
    merge_attr_embedding = Conv2D(16, (3, 3),name='conv')(merge_attr_embedding)
    merge_attr_embedding = BatchNormalization(axis=3,name='normalization')(merge_attr_embedding)#归一化
    #这里加不加最大池化，dense层等需要尝试
    merge_attr_embedding = Activation('relu',name='activation_merge_attr')(merge_attr_embedding)#local_vector
    # merge_attr_embedding = AveragePooling2D((2, 2))(merge_attr_embedding)
    merge_attr_embedding = Dropout(0.35)(merge_attr_embedding)

    # merge_attr_embedding = Conv2D(32, (3, 3))(merge_attr_embedding)
    # merge_attr_embedding = BatchNormalization(axis=3)(merge_attr_embedding)
    # merge_attr_embedding = Activation('relu')(merge_attr_embedding)
    # merge_attr_embedding = MaxPooling2D((2, 2))(merge_attr_embedding)

    # merge_attr_embedding = Conv2D(8, (3, 3))(merge_attr_embedding)
    # merge_attr_embedding = BatchNormalization(axis=3)(merge_attr_embedding)
    # merge_attr_embedding = Activation('relu')(merge_attr_embedding)

    merge_attr_embedding = Flatten(name='flatten_merge_attr')(merge_attr_embedding)
    merge_attr_embedding = merge([merge_attr_embedding, merge_attr_embedding_global], mode='concat',name='merge')

    attr_1 = Dense(32,name='dense_attr_1')(merge_attr_embedding)#16隐藏节点的个数需要调
    attr_1=BatchNormalization()(attr_1)#是否加归一化需要尝试
    attr_1 = Activation('relu',name='activation_attr1')(attr_1)
    #    attr_1=Dropout(0.2)(attr_1)

    # attr_2 = Dense(16)(attr_1)
    # attr_2 = Activation('relu')(attr_2)
    #    id_2=BatchNormalization()(id_2)
    #    id_2=Dropout(0.2)(id_2)

    ########################   id side   ##################################

    user_id_input = Input(shape=(1,), dtype='float32', name='user_id_input')
    user_id_Embedding = Embedding(input_dim=num_users, output_dim=32, name='user_id_Embedding',
                                  embeddings_initializer=RandomNormal(
                                      mean=0.0, stddev=0.01, seed=None),
                                  W_regularizer=l2(0), input_length=1)
    user_id_Embedding = Flatten(name='flatten_user_id')(user_id_Embedding(user_id_input))

    item_id_input = Input(shape=(1,), dtype='float32', name='item_id_input')
    item_id_Embedding = Embedding(input_dim=num_items, output_dim=32, name='item_id_Embedding',
                                  embeddings_initializer=RandomNormal(
                                      mean=0.0, stddev=0.01, seed=None),
                                  W_regularizer=l2(0), input_length=1)
    item_id_Embedding = Flatten(name='flatten_item_id')(item_id_Embedding(item_id_input))

    # id merge embedding
    merge_id_embedding = merge([user_id_Embedding, item_id_Embedding], mode='mul',name='merge_id')
    # id_1 = Dense(64)(merge_id_embedding)
    # id_1 = Activation('relu')(id_1)
    #MLP部分，几层还需要尝试
    id_2 = Dense(32,name='dense_id_2')(merge_id_embedding)
    id_2 = BatchNormalization()(id_2)
    id_2 = Activation('relu',name='activation_id_2')(id_2)

    # merge attr_id embedding
    merge_attr_id_embedding = merge([attr_1, id_2], mode='concat',name='merge_attr_id')
    dense_1 = Dense(64,name='dense_merge_attr_id')(merge_attr_id_embedding)
    dense_1 = BatchNormalization()(dense_1)
    dense_1 = Activation('relu',name='activation')(dense_1)
    # dense_1=BatchNormalization()(dense_1)
    #    dense_1=Dropout(0.2)(dense_1)

    # dense_2=Dense(16)(dense_1)
    # dense_2=Activation('relu')(dense_2)
    #    dense_2=BatchNormalization()(dense_2)
    #    dense_2=Dropout(0.2)(dense_2)

    # dense_3=Dense(8)(dense_2)
    # dense_3=Activation('relu')(dense_3)
    #    dense_3=BatchNormalization()(dense_3)
    #    dense_3=Dropout(0.2)(dense_3)

    # topLayer = Dense(1, activation='sigmoid', init='lecun_uniform',
    #                  name='topLayer')(dense_1)
    topLayer = Dense(1,  init='lecun_uniform',
                     name='topLayer')(dense_1)
    topLayer = BatchNormalization()(topLayer)
    topLayer = Activation(activation='sigmoid')(topLayer)

    # Final prediction layer
    model = Model(input=[user_fea_input, item_fea_input, user_id_input, item_id_input],
                  output=topLayer)
    # if emb_path:#None
    #     from w2vEmbReader import W2VEmbReader as EmbReader
    #     emb_reader = EmbReader(emb_path, emb_dim=emb_size)
    #     print('Initializing word embedding matrix')
    #     model.get_layer('word_emb').set_weights(emb_reader.get_emb_matrix_given_vocab(vocab, model.get_layer('word_emb').get_weights()))
    return model

def get_model4(num_users, num_items,max_seq_len):
    ########################   attr side   ##################################

    # Input
    user_fea_input = Input(shape=(max_seq_len,), dtype='float32', name='user_fea_input')
    # word_emb = Embedding(vocab_size, emb_size, name='word_emb')
    # user_attr_embedding=Flatten(name='flatten_user_fea')(word_emb(user_fea_input))#(?,200,100)
    user_attr_embedding = Dense(16, name='dense_user_fea')(user_fea_input)#(?,200,8)
    user_attr_embedding = BatchNormalization()(user_attr_embedding)
    user_attr_embedding = Activation(activation='relu')(user_attr_embedding)
    user_attr_embedding = Reshape((1,16),name='reshape_user_fea')(user_attr_embedding)

    item_fea_input = Input(shape=(max_seq_len,), dtype='float32', name='item_fea_input')
    # item_attr_embedding=Flatten(name='flatten_item_fea')(word_emb(item_fea_input))
    item_attr_embedding = Dense(16,name='dense_item_fea')(item_fea_input)
    item_attr_embedding = BatchNormalization()(item_attr_embedding)
    item_attr_embedding = Activation(activation='relu')(item_attr_embedding)
    item_attr_embedding = Reshape((16, 1),name='reshape_item_fea')(item_attr_embedding)

    # merge_attr_embedding=Multiply()(user_fea_input,item_fea_input)
    # merge_attr_embedding = merge([user_fea_input, item_fea_input], mode='mul')  # element-wise multiply

    merge_attr_embedding = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[1, 2]),name='lambda_merge')(
        [user_attr_embedding, item_attr_embedding])

    merge_attr_embedding_global = Flatten(name='flatten_merge_attr_global')(merge_attr_embedding)

    merge_attr_embedding = Reshape((16, 16, 1),name='reshape_merge_attr')(merge_attr_embedding)
#用一层卷积还是多层卷积需要尝试
    merge_attr_embedding = Conv2D(16, (3, 3),name='conv')(merge_attr_embedding)
    merge_attr_embedding = BatchNormalization(axis=3,name='normalization')(merge_attr_embedding)#归一化
    #这里加不加最大池化，dense层等需要尝试
    merge_attr_embedding = Activation('relu',name='activation_merge_attr')(merge_attr_embedding)#local_vector
    # merge_attr_embedding = AveragePooling2D((2, 2))(merge_attr_embedding)
    merge_attr_embedding = Dropout(0.5)(merge_attr_embedding)

    # merge_attr_embedding = Conv2D(32, (3, 3))(merge_attr_embedding)
    # merge_attr_embedding = BatchNormalization(axis=3)(merge_attr_embedding)
    # merge_attr_embedding = Activation('relu')(merge_attr_embedding)
    # merge_attr_embedding = MaxPooling2D((2, 2))(merge_attr_embedding)

    # merge_attr_embedding = Conv2D(8, (3, 3))(merge_attr_embedding)
    # merge_attr_embedding = BatchNormalization(axis=3)(merge_attr_embedding)
    # merge_attr_embedding = Activation('relu')(merge_attr_embedding)

    merge_attr_embedding = Flatten(name='flatten_merge_attr')(merge_attr_embedding)
    merge_attr_embedding = merge([merge_attr_embedding, merge_attr_embedding_global], mode='concat',name='merge')

    attr_1 = Dense(32,name='dense_attr_1')(merge_attr_embedding)#16隐藏节点的个数需要调
    attr_1=BatchNormalization()(attr_1)#是否加归一化需要尝试
    attr_1 = Activation('relu',name='activation_attr1')(attr_1)
    #    attr_1=Dropout(0.2)(attr_1)

    # attr_2 = Dense(16)(attr_1)
    # attr_2 = Activation('relu')(attr_2)
    #    id_2=BatchNormalization()(id_2)
    #    id_2=Dropout(0.2)(id_2)

    ########################   id side   ##################################

    user_id_input = Input(shape=(1,), dtype='float32', name='user_id_input')
    user_id_Embedding = Embedding(input_dim=num_users, output_dim=32, name='user_id_Embedding',
                                  embeddings_initializer=RandomNormal(
                                      mean=0.0, stddev=0.01, seed=None),
                                  W_regularizer=l2(0), input_length=1)
    user_id_Embedding = Flatten(name='flatten_user_id')(user_id_Embedding(user_id_input))

    item_id_input = Input(shape=(1,), dtype='float32', name='item_id_input')
    item_id_Embedding = Embedding(input_dim=num_items, output_dim=32, name='item_id_Embedding',
                                  embeddings_initializer=RandomNormal(
                                      mean=0.0, stddev=0.01, seed=None),
                                  W_regularizer=l2(0), input_length=1)
    item_id_Embedding = Flatten(name='flatten_item_id')(item_id_Embedding(item_id_input))

    # id merge embedding
    merge_id_embedding = merge([user_id_Embedding, item_id_Embedding], mode='mul',name='merge_id')
    # id_1 = Dense(64)(merge_id_embedding)
    # id_1 = Activation('relu')(id_1)
    #MLP部分，几层还需要尝试
    id_2 = Dense(32,name='dense_id_2')(merge_id_embedding)
    id_2 = BatchNormalization()(id_2)
    id_2 = Activation('relu',name='activation_id_2')(id_2)

    # merge attr_id embedding
    merge_attr_id_embedding = merge([attr_1, id_2], mode='concat',name='merge_attr_id')
    dense_1 = Dense(64,name='dense_merge_attr_id')(merge_attr_id_embedding)
    dense_1 = BatchNormalization()(dense_1)
    dense_1 = Activation('relu',name='activation')(dense_1)
    # dense_1=BatchNormalization()(dense_1)
    #    dense_1=Dropout(0.2)(dense_1)

    # dense_2=Dense(16)(dense_1)
    # dense_2=Activation('relu')(dense_2)
    #    dense_2=BatchNormalization()(dense_2)
    #    dense_2=Dropout(0.2)(dense_2)

    # dense_3=Dense(8)(dense_2)
    # dense_3=Activation('relu')(dense_3)
    #    dense_3=BatchNormalization()(dense_3)
    #    dense_3=Dropout(0.2)(dense_3)

    # topLayer = Dense(1, activation='sigmoid', init='lecun_uniform',
    #                  name='topLayer')(dense_1)
    topLayer = Dense(1,  init='lecun_uniform',
                     name='topLayer')(dense_1)
    topLayer = BatchNormalization()(topLayer)
    topLayer = Activation(activation='sigmoid')(topLayer)

    # Final prediction layer
    model = Model(input=[user_fea_input, item_fea_input, user_id_input, item_id_input],
                  output=topLayer)
    # if emb_path:#None
    #     from w2vEmbReader import W2VEmbReader as EmbReader
    #     emb_reader = EmbReader(emb_path, emb_dim=emb_size)
    #     print('Initializing word embedding matrix')
    #     model.get_layer('word_emb').set_weights(emb_reader.get_emb_matrix_given_vocab(vocab, model.get_layer('word_emb').get_weights()))
    return model

def get_model5(num_users, num_items,max_seq_len):
    ########################   attr side   ##################################

    # Input
    user_fea_input = Input(shape=(max_seq_len,), dtype='float32', name='user_fea_input')
    # word_emb = Embedding(vocab_size, emb_size, name='word_emb')
    # user_attr_embedding=Flatten(name='flatten_user_fea')(word_emb(user_fea_input))#(?,200,100)
    user_attr_embedding = Dense(32, name='dense_user_fea')(user_fea_input)#(?,200,8)
    user_attr_embedding = BatchNormalization()(user_attr_embedding)
    user_attr_embedding = Activation(activation='relu')(user_attr_embedding)
    user_attr_embedding = Reshape((1,32),name='reshape_user_fea')(user_attr_embedding)

    item_fea_input = Input(shape=(max_seq_len,), dtype='float32', name='item_fea_input')
    # item_attr_embedding=Flatten(name='flatten_item_fea')(word_emb(item_fea_input))
    item_attr_embedding = Dense(32,name='dense_item_fea')(item_fea_input)
    item_attr_embedding = BatchNormalization()(item_attr_embedding)
    item_attr_embedding = Activation(activation='relu')(item_attr_embedding)
    item_attr_embedding = Reshape((32, 1),name='reshape_item_fea')(item_attr_embedding)

    # merge_attr_embedding=Multiply()(user_fea_input,item_fea_input)
    # merge_attr_embedding = merge([user_fea_input, item_fea_input], mode='mul')  # element-wise multiply

    merge_attr_embedding = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[1, 2]),name='lambda_merge')(
        [user_attr_embedding, item_attr_embedding])

    merge_attr_embedding_global = Flatten(name='flatten_merge_attr_global')(merge_attr_embedding)

    merge_attr_embedding = Reshape((32, 32, 1),name='reshape_merge_attr')(merge_attr_embedding)
#用一层卷积还是多层卷积需要尝试
    merge_attr_embedding = Conv2D(32, (3, 3),name='conv')(merge_attr_embedding)
    merge_attr_embedding = BatchNormalization(axis=3,name='normalization')(merge_attr_embedding)#归一化
    #这里加不加最大池化，dense层等需要尝试
    merge_attr_embedding = Activation('relu',name='activation_merge_attr')(merge_attr_embedding)#local_vector
    # merge_attr_embedding = AveragePooling2D((2, 2))(merge_attr_embedding)
    merge_attr_embedding = Dropout(0.5)(merge_attr_embedding)

    # merge_attr_embedding = Conv2D(32, (3, 3))(merge_attr_embedding)
    # merge_attr_embedding = BatchNormalization(axis=3)(merge_attr_embedding)
    # merge_attr_embedding = Activation('relu')(merge_attr_embedding)
    # merge_attr_embedding = MaxPooling2D((2, 2))(merge_attr_embedding)

    # merge_attr_embedding = Conv2D(8, (3, 3))(merge_attr_embedding)
    # merge_attr_embedding = BatchNormalization(axis=3)(merge_attr_embedding)
    # merge_attr_embedding = Activation('relu')(merge_attr_embedding)

    merge_attr_embedding = Flatten(name='flatten_merge_attr')(merge_attr_embedding)
    merge_attr_embedding = merge([merge_attr_embedding, merge_attr_embedding_global], mode='concat',name='merge')

    attr_1 = Dense(32,name='dense_attr_1')(merge_attr_embedding)#16隐藏节点的个数需要调
    attr_1=BatchNormalization()(attr_1)#是否加归一化需要尝试
    attr_1 = Activation('relu',name='activation_attr1')(attr_1)
    attr_1=Dropout(0.5)(attr_1)

    # attr_2 = Dense(16)(attr_1)
    # attr_2 = Activation('relu')(attr_2)
    #    id_2=BatchNormalization()(id_2)
    #    id_2=Dropout(0.2)(id_2)

    ########################   id side   ##################################

    user_id_input = Input(shape=(1,), dtype='float32', name='user_id_input')
    user_id_Embedding = Embedding(input_dim=num_users, output_dim=32, name='user_id_Embedding',
                                  embeddings_initializer=RandomNormal(
                                      mean=0.0, stddev=0.01, seed=None),
                                  W_regularizer=l2(0), input_length=1)
    user_id_Embedding = Flatten(name='flatten_user_id')(user_id_Embedding(user_id_input))

    item_id_input = Input(shape=(1,), dtype='float32', name='item_id_input')
    item_id_Embedding = Embedding(input_dim=num_items, output_dim=32, name='item_id_Embedding',
                                  embeddings_initializer=RandomNormal(
                                      mean=0.0, stddev=0.01, seed=None),
                                  W_regularizer=l2(0), input_length=1)
    item_id_Embedding = Flatten(name='flatten_item_id')(item_id_Embedding(item_id_input))

    # id merge embedding
    merge_id_embedding = merge([user_id_Embedding, item_id_Embedding], mode='mul',name='merge_id')
    # id_1 = Dense(64)(merge_id_embedding)
    # id_1 = Activation('relu')(id_1)
    #MLP部分，几层还需要尝试
    id_2 = Dense(32,name='dense_id_2')(merge_id_embedding)
    id_2 = BatchNormalization()(id_2)
    id_2 = Activation('relu',name='activation_id_2')(id_2)
    id_2 = Dropout(0.5)(id_2)

    # merge attr_id embedding
    merge_attr_id_embedding = merge([attr_1, id_2], mode='concat',name='merge_attr_id')
    dense_1 = Dense(64,name='dense_merge_attr_id')(merge_attr_id_embedding)
    dense_1 = BatchNormalization()(dense_1)
    dense_1 = Activation('relu',name='activation')(dense_1)
    dense_1 = Dropout(0.5)(dense_1)
    # dense_1=BatchNormalization()(dense_1)
    #    dense_1=Dropout(0.2)(dense_1)

    # dense_2=Dense(16)(dense_1)
    # dense_2=Activation('relu')(dense_2)
    #    dense_2=BatchNormalization()(dense_2)
    #    dense_2=Dropout(0.2)(dense_2)

    # dense_3=Dense(8)(dense_2)
    # dense_3=Activation('relu')(dense_3)
    #    dense_3=BatchNormalization()(dense_3)
    #    dense_3=Dropout(0.2)(dense_3)

    # topLayer = Dense(1, activation='sigmoid', init='lecun_uniform',
    #                  name='topLayer')(dense_1)
    topLayer = Dense(1,  init='lecun_uniform',
                     name='topLayer')(dense_1)
    topLayer = BatchNormalization()(topLayer)
    topLayer = Activation(activation='sigmoid')(topLayer)

    # Final prediction layer
    model = Model(input=[user_fea_input, item_fea_input, user_id_input, item_id_input],
                  output=topLayer)
    return model


def get_model6(num_users, num_items,max_seq_len):
    ########################   attr side   ##################################

    # Input
    user_fea_input = Input(shape=(max_seq_len,), dtype='float32', name='user_fea_input')
    # word_emb = Embedding(vocab_size, emb_size, name='word_emb')
    # user_attr_embedding=Flatten(name='flatten_user_fea')(word_emb(user_fea_input))#(?,200,100)
    user_attr_embedding = Dense(32, name='dense_user_fea')(user_fea_input)#(?,200,8)
    user_attr_embedding = BatchNormalization()(user_attr_embedding)
    user_attr_embedding = Activation(activation='relu')(user_attr_embedding)
    user_attr_embedding = Reshape((1,32),name='reshape_user_fea')(user_attr_embedding)

    item_fea_input = Input(shape=(max_seq_len,), dtype='float32', name='item_fea_input')
    # item_attr_embedding=Flatten(name='flatten_item_fea')(word_emb(item_fea_input))
    item_attr_embedding = Dense(32,name='dense_item_fea')(item_fea_input)
    item_attr_embedding = BatchNormalization()(item_attr_embedding)
    item_attr_embedding = Activation(activation='relu')(item_attr_embedding)
    item_attr_embedding = Reshape((32, 1),name='reshape_item_fea')(item_attr_embedding)

    # merge_attr_embedding=Multiply()(user_fea_input,item_fea_input)
    # merge_attr_embedding = merge([user_fea_input, item_fea_input], mode='mul')  # element-wise multiply

    merge_attr_embedding = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[1, 2]),name='lambda_merge')(
        [user_attr_embedding, item_attr_embedding])

    merge_attr_embedding_global = Flatten(name='flatten_merge_attr_global')(merge_attr_embedding)

    merge_attr_embedding = Reshape((32, 32, 1),name='reshape_merge_attr')(merge_attr_embedding)
#用一层卷积还是多层卷积需要尝试
    merge_attr_embedding = Conv2D(32, (3, 3),name='conv')(merge_attr_embedding)
    merge_attr_embedding = BatchNormalization(axis=3,name='normalization')(merge_attr_embedding)#归一化
    #这里加不加最大池化，dense层等需要尝试
    merge_attr_embedding = Activation('relu',name='activation_merge_attr')(merge_attr_embedding)#local_vector
    # merge_attr_embedding = AveragePooling2D((2, 2))(merge_attr_embedding)
    merge_attr_embedding = Dropout(0.7)(merge_attr_embedding)

    # merge_attr_embedding = Conv2D(32, (3, 3))(merge_attr_embedding)
    # merge_attr_embedding = BatchNormalization(axis=3)(merge_attr_embedding)
    # merge_attr_embedding = Activation('relu')(merge_attr_embedding)
    # merge_attr_embedding = MaxPooling2D((2, 2))(merge_attr_embedding)

    # merge_attr_embedding = Conv2D(8, (3, 3))(merge_attr_embedding)
    # merge_attr_embedding = BatchNormalization(axis=3)(merge_attr_embedding)
    # merge_attr_embedding = Activation('relu')(merge_attr_embedding)

    merge_attr_embedding = Flatten(name='flatten_merge_attr')(merge_attr_embedding)
    merge_attr_embedding = merge([merge_attr_embedding, merge_attr_embedding_global], mode='concat',name='merge')

    attr_1 = Dense(32,name='dense_attr_1')(merge_attr_embedding)#16隐藏节点的个数需要调
    attr_1=BatchNormalization()(attr_1)#是否加归一化需要尝试
    attr_1 = Activation('relu',name='activation_attr1')(attr_1)
    attr_1=Dropout(0.7)(attr_1)

    # attr_2 = Dense(16)(attr_1)
    # attr_2 = Activation('relu')(attr_2)
    #    id_2=BatchNormalization()(id_2)
    #    id_2=Dropout(0.2)(id_2)

    ########################   id side   ##################################

    user_id_input = Input(shape=(1,), dtype='float32', name='user_id_input')
    user_id_Embedding = Embedding(input_dim=num_users, output_dim=32, name='user_id_Embedding',
                                  embeddings_initializer=RandomNormal(
                                      mean=0.0, stddev=0.01, seed=None),
                                  W_regularizer=l2(0), input_length=1)
    user_id_Embedding = Flatten(name='flatten_user_id')(user_id_Embedding(user_id_input))

    item_id_input = Input(shape=(1,), dtype='float32', name='item_id_input')
    item_id_Embedding = Embedding(input_dim=num_items, output_dim=32, name='item_id_Embedding',
                                  embeddings_initializer=RandomNormal(
                                      mean=0.0, stddev=0.01, seed=None),
                                  W_regularizer=l2(0), input_length=1)
    item_id_Embedding = Flatten(name='flatten_item_id')(item_id_Embedding(item_id_input))

    # id merge embedding
    merge_id_embedding = merge([user_id_Embedding, item_id_Embedding], mode='mul',name='merge_id')
    # id_1 = Dense(64)(merge_id_embedding)
    # id_1 = Activation('relu')(id_1)
    #MLP部分，几层还需要尝试
    id_2 = Dense(32,name='dense_id_2')(merge_id_embedding)
    id_2 = BatchNormalization()(id_2)
    id_2 = Activation('relu',name='activation_id_2')(id_2)
    id_2 = Dropout(0.7)(id_2)

    # merge attr_id embedding
    merge_attr_id_embedding = merge([attr_1, id_2], mode='concat',name='merge_attr_id')
    dense_1 = Dense(64,name='dense_merge_attr_id')(merge_attr_id_embedding)
    dense_1 = BatchNormalization()(dense_1)
    dense_1 = Activation('relu',name='activation')(dense_1)
    dense_1 = Dropout(0.7)(dense_1)
    # dense_1=BatchNormalization()(dense_1)
    #    dense_1=Dropout(0.2)(dense_1)

    # dense_2=Dense(16)(dense_1)
    # dense_2=Activation('relu')(dense_2)
    #    dense_2=BatchNormalization()(dense_2)
    #    dense_2=Dropout(0.2)(dense_2)

    # dense_3=Dense(8)(dense_2)
    # dense_3=Activation('relu')(dense_3)
    #    dense_3=BatchNormalization()(dense_3)
    #    dense_3=Dropout(0.2)(dense_3)

    # topLayer = Dense(1, activation='sigmoid', init='lecun_uniform',
    #                  name='topLayer')(dense_1)
    topLayer = Dense(1,  init='lecun_uniform',
                     name='topLayer')(dense_1)
    topLayer = BatchNormalization()(topLayer)
    topLayer = Activation(activation='sigmoid')(topLayer)
    # topLayer = Dropout(0.5)(topLayer)

    # Final prediction layer
    model = Model(input=[user_fea_input, item_fea_input, user_id_input, item_id_input],
                  output=topLayer)
    return model

def get_model7(num_users, num_items,max_seq_len):
    ########################   attr side   ##################################

    # Input
    user_fea_input = Input(shape=(max_seq_len,), dtype='float32', name='user_fea_input')
    # word_emb = Embedding(vocab_size, emb_size, name='word_emb')
    # user_attr_embedding=Flatten(name='flatten_user_fea')(word_emb(user_fea_input))#(?,200,100)
    user_attr_embedding = Dense(64, name='dense_user_fea')(user_fea_input)#(?,200,8)
    user_attr_embedding = BatchNormalization()(user_attr_embedding)
    user_attr_embedding = Activation(activation='relu')(user_attr_embedding)
    user_attr_embedding = Reshape((1,64),name='reshape_user_fea')(user_attr_embedding)

    item_fea_input = Input(shape=(max_seq_len,), dtype='float32', name='item_fea_input')
    # item_attr_embedding=Flatten(name='flatten_item_fea')(word_emb(item_fea_input))
    item_attr_embedding = Dense(64,name='dense_item_fea')(item_fea_input)
    item_attr_embedding = BatchNormalization()(item_attr_embedding)
    item_attr_embedding = Activation(activation='relu')(item_attr_embedding)
    item_attr_embedding = Reshape((64, 1),name='reshape_item_fea')(item_attr_embedding)

    # merge_attr_embedding=Multiply()(user_fea_input,item_fea_input)
    # merge_attr_embedding = merge([user_fea_input, item_fea_input], mode='mul')  # element-wise multiply

    merge_attr_embedding = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[1, 2]),name='lambda_merge')(
        [user_attr_embedding, item_attr_embedding])

    merge_attr_embedding_global = Flatten(name='flatten_merge_attr_global')(merge_attr_embedding)

    merge_attr_embedding = Reshape((64, 64, 1),name='reshape_merge_attr')(merge_attr_embedding)
#用一层卷积还是多层卷积需要尝试
    merge_attr_embedding = Conv2D(16, (3, 3),name='conv')(merge_attr_embedding)
    merge_attr_embedding = BatchNormalization(axis=3,name='normalization')(merge_attr_embedding)#归一化
    #这里加不加最大池化，dense层等需要尝试
    merge_attr_embedding = Activation('relu',name='activation_merge_attr')(merge_attr_embedding)#local_vector
    merge_attr_embedding = MaxPooling2D((2, 2))(merge_attr_embedding)
    merge_attr_embedding = Dropout(0.7)(merge_attr_embedding)

    # merge_attr_embedding = Conv2D(32, (3, 3))(merge_attr_embedding)
    # merge_attr_embedding = BatchNormalization(axis=3)(merge_attr_embedding)
    # merge_attr_embedding = Activation('relu')(merge_attr_embedding)
    # merge_attr_embedding = MaxPooling2D((2, 2))(merge_attr_embedding)

    # merge_attr_embedding = Conv2D(8, (3, 3))(merge_attr_embedding)
    # merge_attr_embedding = BatchNormalization(axis=3)(merge_attr_embedding)
    # merge_attr_embedding = Activation('relu')(merge_attr_embedding)

    merge_attr_embedding = Flatten(name='flatten_merge_attr')(merge_attr_embedding)
    merge_attr_embedding = merge([merge_attr_embedding, merge_attr_embedding_global], mode='concat',name='merge')

    attr_1 = Dense(64,name='dense_attr_1')(merge_attr_embedding)#16隐藏节点的个数需要调
    attr_1=BatchNormalization()(attr_1)#是否加归一化需要尝试
    attr_1 = Activation('relu',name='activation_attr1')(attr_1)
    attr_1=Dropout(0.5)(attr_1)

    # attr_2 = Dense(16)(attr_1)
    # attr_2 = Activation('relu')(attr_2)
    #    id_2=BatchNormalization()(id_2)
    #    id_2=Dropout(0.2)(id_2)

    ########################   id side   ##################################

    user_id_input = Input(shape=(1,), dtype='float32', name='user_id_input')
    user_id_Embedding = Embedding(input_dim=num_users, output_dim=32, name='user_id_Embedding',
                                  embeddings_initializer=RandomNormal(
                                      mean=0.0, stddev=0.01, seed=None),
                                  W_regularizer=l2(0), input_length=1)
    user_id_Embedding = Flatten(name='flatten_user_id')(user_id_Embedding(user_id_input))

    item_id_input = Input(shape=(1,), dtype='float32', name='item_id_input')
    item_id_Embedding = Embedding(input_dim=num_items, output_dim=32, name='item_id_Embedding',
                                  embeddings_initializer=RandomNormal(
                                      mean=0.0, stddev=0.01, seed=None),
                                  W_regularizer=l2(0), input_length=1)
    item_id_Embedding = Flatten(name='flatten_item_id')(item_id_Embedding(item_id_input))

    # id merge embedding
    merge_id_embedding = merge([user_id_Embedding, item_id_Embedding], mode='mul',name='merge_id')
    # id_1 = Dense(64)(merge_id_embedding)
    # id_1 = Activation('relu')(id_1)
    #MLP部分，几层还需要尝试
    id_2 = Dense(64,name='dense_id_2')(merge_id_embedding)
    id_2 = BatchNormalization()(id_2)
    id_2 = Activation('relu',name='activation_id_2')(id_2)
    id_2 = Dropout(0.5)(id_2)

    # merge attr_id embedding
    merge_attr_id_embedding = merge([attr_1, id_2], mode='concat',name='merge_attr_id')
    dense_1 = Dense(64,name='dense_merge_attr_id')(merge_attr_id_embedding)
    dense_1 = BatchNormalization()(dense_1)
    dense_1 = Activation('relu',name='activation')(dense_1)
    dense_1 = Dropout(0.5)(dense_1)
    # dense_1=BatchNormalization()(dense_1)
    #    dense_1=Dropout(0.2)(dense_1)

    # dense_2=Dense(16)(dense_1)
    # dense_2=Activation('relu')(dense_2)
    #    dense_2=BatchNormalization()(dense_2)
    #    dense_2=Dropout(0.2)(dense_2)

    # dense_3=Dense(8)(dense_2)
    # dense_3=Activation('relu')(dense_3)
    #    dense_3=BatchNormalization()(dense_3)
    #    dense_3=Dropout(0.2)(dense_3)

    # topLayer = Dense(1, activation='sigmoid', init='lecun_uniform',
    #                  name='topLayer')(dense_1)
    topLayer = Dense(1,  init='lecun_uniform',
                     name='topLayer')(dense_1)
    topLayer = BatchNormalization()(topLayer)
    topLayer = Activation(activation='sigmoid')(topLayer)

    # Final prediction layer
    model = Model(input=[user_fea_input, item_fea_input, user_id_input, item_id_input],
                  output=topLayer)
    return model


def get_model8(num_users, num_items,max_seq_len):
    ########################   attr side   ##################################

    # Input
    user_fea_input = Input(shape=(max_seq_len,), dtype='float32', name='user_fea_input')
    # word_emb = Embedding(vocab_size, emb_size, name='word_emb')
    # user_attr_embedding=Flatten(name='flatten_user_fea')(word_emb(user_fea_input))#(?,200,100)
    user_attr_embedding = Dense(64, name='dense_user_fea')(user_fea_input)#(?,200,8)
    user_attr_embedding = BatchNormalization()(user_attr_embedding)
    user_attr_embedding = Activation(activation='relu')(user_attr_embedding)
    user_attr_embedding = Dropout(0.3)(user_attr_embedding)
    user_attr_embedding = Reshape((1,64),name='reshape_user_fea')(user_attr_embedding)

    item_fea_input = Input(shape=(max_seq_len,), dtype='float32', name='item_fea_input')
    # item_attr_embedding=Flatten(name='flatten_item_fea')(word_emb(item_fea_input))
    item_attr_embedding = Dense(64,name='dense_item_fea')(item_fea_input)
    item_attr_embedding = BatchNormalization()(item_attr_embedding)
    item_attr_embedding = Activation(activation='relu')(item_attr_embedding)
    item_attr_embedding = Dropout(0.3)(item_attr_embedding)
    item_attr_embedding = Reshape((64, 1),name='reshape_item_fea')(item_attr_embedding)

    # merge_attr_embedding=Multiply()(user_fea_input,item_fea_input)
    # merge_attr_embedding = merge([user_fea_input, item_fea_input], mode='mul')  # element-wise multiply

    merge_attr_embedding = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[1, 2]),name='lambda_merge')(
        [user_attr_embedding, item_attr_embedding])

    merge_attr_embedding_global = Flatten(name='flatten_merge_attr_global')(merge_attr_embedding)

    merge_attr_embedding = Reshape((64, 64, 1),name='reshape_merge_attr')(merge_attr_embedding)
#用一层卷积还是多层卷积需要尝试
    merge_attr_embedding = Conv2D(32, (3, 3),name='conv')(merge_attr_embedding)
    merge_attr_embedding = BatchNormalization(axis=3,name='normalization')(merge_attr_embedding)#归一化
    #这里加不加最大池化，dense层等需要尝试
    merge_attr_embedding = Activation('relu',name='activation_merge_attr')(merge_attr_embedding)#local_vector
    merge_attr_embedding = MaxPooling2D((2, 2))(merge_attr_embedding)
    merge_attr_embedding = Dropout(0.7)(merge_attr_embedding)

    # merge_attr_embedding = Conv2D(32, (3, 3))(merge_attr_embedding)
    # merge_attr_embedding = BatchNormalization(axis=3)(merge_attr_embedding)
    # merge_attr_embedding = Activation('relu')(merge_attr_embedding)
    # merge_attr_embedding = MaxPooling2D((2, 2))(merge_attr_embedding)

    # merge_attr_embedding = Conv2D(8, (3, 3))(merge_attr_embedding)
    # merge_attr_embedding = BatchNormalization(axis=3)(merge_attr_embedding)
    # merge_attr_embedding = Activation('relu')(merge_attr_embedding)

    merge_attr_embedding = Flatten(name='flatten_merge_attr')(merge_attr_embedding)
    merge_attr_embedding = merge([merge_attr_embedding, merge_attr_embedding_global], mode='concat',name='merge')

    attr_1 = Dense(32,name='dense_attr_1')(merge_attr_embedding)#16隐藏节点的个数需要调
    attr_1=BatchNormalization()(attr_1)#是否加归一化需要尝试
    attr_1 = Activation('relu',name='activation_attr1')(attr_1)
    attr_1=Dropout(0.5)(attr_1)

    # attr_2 = Dense(16)(attr_1)
    # attr_2 = Activation('relu')(attr_2)
    #    id_2=BatchNormalization()(id_2)
    #    id_2=Dropout(0.2)(id_2)

    ########################   id side   ##################################

    user_id_input = Input(shape=(1,), dtype='float32', name='user_id_input')
    user_id_Embedding = Embedding(input_dim=num_users, output_dim=32, name='user_id_Embedding',
                                  embeddings_initializer=RandomNormal(
                                      mean=0.0, stddev=0.01, seed=None),
                                  W_regularizer=l2(0), input_length=1)
    user_id_Embedding = Flatten(name='flatten_user_id')(user_id_Embedding(user_id_input))

    item_id_input = Input(shape=(1,), dtype='float32', name='item_id_input')
    item_id_Embedding = Embedding(input_dim=num_items, output_dim=32, name='item_id_Embedding',
                                  embeddings_initializer=RandomNormal(
                                      mean=0.0, stddev=0.01, seed=None),
                                  W_regularizer=l2(0), input_length=1)
    item_id_Embedding = Flatten(name='flatten_item_id')(item_id_Embedding(item_id_input))

    # id merge embedding
    merge_id_embedding = merge([user_id_Embedding, item_id_Embedding], mode='mul',name='merge_id')
    # id_1 = Dense(64)(merge_id_embedding)
    # id_1 = Activation('relu')(id_1)
    #MLP部分，几层还需要尝试
    id_2 = Dense(32,name='dense_id_2')(merge_id_embedding)
    id_2 = BatchNormalization()(id_2)
    id_2 = Activation('relu',name='activation_id_2')(id_2)
    id_2 = Dropout(0.5)(id_2)

    # merge attr_id embedding
    merge_attr_id_embedding = merge([attr_1, id_2], mode='concat',name='merge_attr_id')
    dense_1 = Dense(64,name='dense_merge_attr_id')(merge_attr_id_embedding)
    dense_1 = BatchNormalization()(dense_1)
    dense_1 = Activation('relu',name='activation')(dense_1)
    dense_1 = Dropout(0.5)(dense_1)
    # dense_1=BatchNormalization()(dense_1)
    #    dense_1=Dropout(0.2)(dense_1)

    # dense_2=Dense(16)(dense_1)
    # dense_2=Activation('relu')(dense_2)
    #    dense_2=BatchNormalization()(dense_2)
    #    dense_2=Dropout(0.2)(dense_2)

    # dense_3=Dense(8)(dense_2)
    # dense_3=Activation('relu')(dense_3)
    #    dense_3=BatchNormalization()(dense_3)
    #    dense_3=Dropout(0.2)(dense_3)

    # topLayer = Dense(1, activation='sigmoid', init='lecun_uniform',
    #                  name='topLayer')(dense_1)
    topLayer = Dense(1,  init='lecun_uniform',
                     name='topLayer')(dense_1)
    topLayer = BatchNormalization()(topLayer)
    topLayer = Activation(activation='sigmoid')(topLayer)
    # topLayer = Dropout(0.2)(topLayer)

    # Final prediction layer
    model = Model(input=[user_fea_input, item_fea_input, user_id_input, item_id_input],
                  output=topLayer)
    return model

def get_model9(num_users, num_items,max_seq_len):
    ########################   attr side   ##################################

    # Input
    user_fea_input = Input(shape=(max_seq_len,), dtype='float32', name='user_fea_input')
    # word_emb = Embedding(vocab_size, emb_size, name='word_emb')
    # user_attr_embedding=Flatten(name='flatten_user_fea')(word_emb(user_fea_input))#(?,200,100)
    user_attr_embedding = Dense(64, name='dense_user_fea')(user_fea_input)#(?,200,8)
    user_attr_embedding = BatchNormalization()(user_attr_embedding)
    user_attr_embedding = Activation(activation='relu')(user_attr_embedding)
    # user_attr_embedding = Dropout(0.3)(user_attr_embedding)
    user_attr_embedding = Reshape((1,64),name='reshape_user_fea')(user_attr_embedding)

    item_fea_input = Input(shape=(max_seq_len,), dtype='float32', name='item_fea_input')
    # item_attr_embedding=Flatten(name='flatten_item_fea')(word_emb(item_fea_input))
    item_attr_embedding = Dense(64,name='dense_item_fea')(item_fea_input)
    item_attr_embedding = BatchNormalization()(item_attr_embedding)
    item_attr_embedding = Activation(activation='relu')(item_attr_embedding)
    # item_attr_embedding = Dropout(0.3)(item_attr_embedding)
    item_attr_embedding = Reshape((64, 1),name='reshape_item_fea')(item_attr_embedding)

    # merge_attr_embedding=Multiply()(user_fea_input,item_fea_input)
    # merge_attr_embedding = merge([user_fea_input, item_fea_input], mode='mul')  # element-wise multiply

    merge_attr_embedding = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[1, 2]),name='lambda_merge')(
        [user_attr_embedding, item_attr_embedding])

    merge_attr_embedding_global = Flatten(name='flatten_merge_attr_global')(merge_attr_embedding)

    merge_attr_embedding = Reshape((64, 64, 1),name='reshape_merge_attr')(merge_attr_embedding)
#用一层卷积还是多层卷积需要尝试
    merge_attr_embedding = Conv2D(64, (3, 3),name='conv')(merge_attr_embedding)
    merge_attr_embedding = BatchNormalization(axis=3,name='normalization')(merge_attr_embedding)#归一化
    #这里加不加最大池化，dense层等需要尝试
    merge_attr_embedding = Activation('relu',name='activation_merge_attr')(merge_attr_embedding)#local_vector
    merge_attr_embedding = MaxPooling2D((2, 2))(merge_attr_embedding)
    merge_attr_embedding = Dropout(0.7)(merge_attr_embedding)

    merge_attr_embedding = Conv2D(32, (3, 3), name='conv1')(merge_attr_embedding)
    merge_attr_embedding = BatchNormalization(axis=3, name='normalization1')(merge_attr_embedding)  # 归一化
    # 这里加不加最大池化，dense层等需要尝试
    merge_attr_embedding = Activation('relu', name='activation_merge_attr1')(merge_attr_embedding)  # local_vector
    merge_attr_embedding = MaxPooling2D((2, 2))(merge_attr_embedding)
    merge_attr_embedding = Dropout(0.7)(merge_attr_embedding)

    # merge_attr_embedding = Conv2D(32, (3, 3))(merge_attr_embedding)
    # merge_attr_embedding = BatchNormalization(axis=3)(merge_attr_embedding)
    # merge_attr_embedding = Activation('relu')(merge_attr_embedding)
    # merge_attr_embedding = MaxPooling2D((2, 2))(merge_attr_embedding)

    # merge_attr_embedding = Conv2D(8, (3, 3))(merge_attr_embedding)
    # merge_attr_embedding = BatchNormalization(axis=3)(merge_attr_embedding)
    # merge_attr_embedding = Activation('relu')(merge_attr_embedding)

    merge_attr_embedding = Flatten(name='flatten_merge_attr')(merge_attr_embedding)
    merge_attr_embedding = merge([merge_attr_embedding, merge_attr_embedding_global], mode='concat',name='merge')

    attr_1 = Dense(32,name='dense_attr_1')(merge_attr_embedding)#16隐藏节点的个数需要调
    attr_1=BatchNormalization()(attr_1)#是否加归一化需要尝试
    attr_1 = Activation('relu',name='activation_attr1')(attr_1)
    attr_1=Dropout(0.7)(attr_1)

    # attr_2 = Dense(16)(attr_1)
    # attr_2 = Activation('relu')(attr_2)
    #    id_2=BatchNormalization()(id_2)
    #    id_2=Dropout(0.2)(id_2)

    ########################   id side   ##################################

    user_id_input = Input(shape=(1,), dtype='float32', name='user_id_input')
    user_id_Embedding = Embedding(input_dim=num_users, output_dim=32, name='user_id_Embedding',
                                  embeddings_initializer=RandomNormal(
                                      mean=0.0, stddev=0.01, seed=None),
                                  W_regularizer=l2(0), input_length=1)
    user_id_Embedding = Flatten(name='flatten_user_id')(user_id_Embedding(user_id_input))

    item_id_input = Input(shape=(1,), dtype='float32', name='item_id_input')
    item_id_Embedding = Embedding(input_dim=num_items, output_dim=32, name='item_id_Embedding',
                                  embeddings_initializer=RandomNormal(
                                      mean=0.0, stddev=0.01, seed=None),
                                  W_regularizer=l2(0), input_length=1)
    item_id_Embedding = Flatten(name='flatten_item_id')(item_id_Embedding(item_id_input))

    # id merge embedding
    merge_id_embedding = merge([user_id_Embedding, item_id_Embedding], mode='mul',name='merge_id')
    # id_1 = Dense(64)(merge_id_embedding)
    # id_1 = Activation('relu')(id_1)
    #MLP部分，几层还需要尝试
    id_2 = Dense(32,name='dense_id_2')(merge_id_embedding)
    id_2 = BatchNormalization()(id_2)
    id_2 = Activation('relu',name='activation_id_2')(id_2)
    id_2 = Dropout(0.5)(id_2)

    # merge attr_id embedding
    merge_attr_id_embedding = merge([attr_1, id_2], mode='concat',name='merge_attr_id')
    dense_1 = Dense(64,name='dense_merge_attr_id')(merge_attr_id_embedding)
    dense_1 = BatchNormalization()(dense_1)
    dense_1 = Activation('relu',name='activation')(dense_1)
    dense_1 = Dropout(0.5)(dense_1)
    # dense_1=BatchNormalization()(dense_1)
    #    dense_1=Dropout(0.2)(dense_1)

    # dense_2=Dense(16)(dense_1)
    # dense_2=Activation('relu')(dense_2)
    #    dense_2=BatchNormalization()(dense_2)
    #    dense_2=Dropout(0.2)(dense_2)

    # dense_3=Dense(8)(dense_2)
    # dense_3=Activation('relu')(dense_3)
    #    dense_3=BatchNormalization()(dense_3)
    #    dense_3=Dropout(0.2)(dense_3)

    # topLayer = Dense(1, activation='sigmoid', init='lecun_uniform',
    #                  name='topLayer')(dense_1)
    topLayer = Dense(1,  init='lecun_uniform',
                     name='topLayer')(dense_1)
    topLayer = BatchNormalization()(topLayer)
    topLayer = Activation(activation='sigmoid')(topLayer)
    # topLayer = Dropout(0.2)(topLayer)

    # Final prediction layer
    model = Model(input=[user_fea_input, item_fea_input, user_id_input, item_id_input],
                  output=topLayer)
    return model


def get_model10(num_users, num_items,max_seq_len):
    ########################   attr side   ##################################

    # Input
    user_fea_input = Input(shape=(max_seq_len,), dtype='float32', name='user_fea_input')
    # word_emb = Embedding(vocab_size, emb_size, name='word_emb')
    # user_attr_embedding=Flatten(name='flatten_user_fea')(word_emb(user_fea_input))#(?,200,100)
    user_attr_embedding = Dense(64, name='dense_user_fea')(user_fea_input)#(?,200,8)
    user_attr_embedding = BatchNormalization()(user_attr_embedding)
    user_attr_embedding = Activation(activation='relu')(user_attr_embedding)
    # user_attr_embedding = Dropout(0.3)(user_attr_embedding)
    user_attr_embedding = Reshape((1,64),name='reshape_user_fea')(user_attr_embedding)

    item_fea_input = Input(shape=(max_seq_len,), dtype='float32', name='item_fea_input')
    # item_attr_embedding=Flatten(name='flatten_item_fea')(word_emb(item_fea_input))
    item_attr_embedding = Dense(64,name='dense_item_fea')(item_fea_input)
    item_attr_embedding = BatchNormalization()(item_attr_embedding)
    item_attr_embedding = Activation(activation='relu')(item_attr_embedding)
    # item_attr_embedding = Dropout(0.3)(item_attr_embedding)
    item_attr_embedding = Reshape((64, 1),name='reshape_item_fea')(item_attr_embedding)

    # merge_attr_embedding=Multiply()(user_fea_input,item_fea_input)
    # merge_attr_embedding = merge([user_fea_input, item_fea_input], mode='mul')  # element-wise multiply

    merge_attr_embedding = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[1, 2]),name='lambda_merge')(
        [user_attr_embedding, item_attr_embedding])

    merge_attr_embedding_global = Flatten(name='flatten_merge_attr_global')(merge_attr_embedding)

    merge_attr_embedding = Reshape((64, 64, 1),name='reshape_merge_attr')(merge_attr_embedding)
#用一层卷积还是多层卷积需要尝试
    merge_attr_embedding = Conv2D(64, (3, 3),name='conv')(merge_attr_embedding)
    merge_attr_embedding = BatchNormalization(axis=3,name='normalization')(merge_attr_embedding)#归一化
    #这里加不加最大池化，dense层等需要尝试
    merge_attr_embedding = Activation('relu',name='activation_merge_attr')(merge_attr_embedding)#local_vector
    merge_attr_embedding = MaxPooling2D((2, 2))(merge_attr_embedding)
    merge_attr_embedding = Dropout(0.7)(merge_attr_embedding)

    merge_attr_embedding = Conv2D(32, (3, 3), name='conv1')(merge_attr_embedding)
    merge_attr_embedding = BatchNormalization(axis=3, name='normalization1')(merge_attr_embedding)  # 归一化
    # 这里加不加最大池化，dense层等需要尝试
    merge_attr_embedding = Activation('relu', name='activation_merge_attr1')(merge_attr_embedding)  # local_vector
    merge_attr_embedding = MaxPooling2D((2, 2))(merge_attr_embedding)
    merge_attr_embedding = Dropout(0.7)(merge_attr_embedding)

    # merge_attr_embedding = Conv2D(32, (3, 3))(merge_attr_embedding)
    # merge_attr_embedding = BatchNormalization(axis=3)(merge_attr_embedding)
    # merge_attr_embedding = Activation('relu')(merge_attr_embedding)
    # merge_attr_embedding = MaxPooling2D((2, 2))(merge_attr_embedding)

    # merge_attr_embedding = Conv2D(8, (3, 3))(merge_attr_embedding)
    # merge_attr_embedding = BatchNormalization(axis=3)(merge_attr_embedding)
    # merge_attr_embedding = Activation('relu')(merge_attr_embedding)

    merge_attr_embedding = Flatten(name='flatten_merge_attr')(merge_attr_embedding)
    merge_attr_embedding = merge([merge_attr_embedding, merge_attr_embedding_global], mode='concat',name='merge')

    attr_1 = Dense(32,name='dense_attr_1')(merge_attr_embedding)#16隐藏节点的个数需要调
    attr_1=BatchNormalization()(attr_1)#是否加归一化需要尝试
    attr_1 = Activation('relu',name='activation_attr1')(attr_1)
    attr_1=Dropout(0.2)(attr_1)

    # attr_2 = Dense(16)(attr_1)
    # attr_2 = Activation('relu')(attr_2)
    #    id_2=BatchNormalization()(id_2)
    #    id_2=Dropout(0.2)(id_2)

    ########################   id side   ##################################

    user_id_input = Input(shape=(1,), dtype='float32', name='user_id_input')
    user_id_Embedding = Embedding(input_dim=num_users, output_dim=32, name='user_id_Embedding',
                                  embeddings_initializer=RandomNormal(
                                      mean=0.0, stddev=0.01, seed=None),
                                  W_regularizer=l2(0), input_length=1)
    user_id_Embedding = Flatten(name='flatten_user_id')(user_id_Embedding(user_id_input))

    item_id_input = Input(shape=(1,), dtype='float32', name='item_id_input')
    item_id_Embedding = Embedding(input_dim=num_items, output_dim=32, name='item_id_Embedding',
                                  embeddings_initializer=RandomNormal(
                                      mean=0.0, stddev=0.01, seed=None),
                                  W_regularizer=l2(0), input_length=1)
    item_id_Embedding = Flatten(name='flatten_item_id')(item_id_Embedding(item_id_input))

    # id merge embedding
    merge_id_embedding = merge([user_id_Embedding, item_id_Embedding], mode='mul',name='merge_id')
    # id_1 = Dense(64)(merge_id_embedding)
    # id_1 = Activation('relu')(id_1)
    #MLP部分，几层还需要尝试
    id_2 = Dense(64,name='dense_id_2')(merge_id_embedding)
    id_2 = BatchNormalization()(id_2)
    id_2 = Activation('relu',name='activation_id_2')(id_2)
    id_2 = Dropout(0.5)(id_2)

    id_2 = Dense(32, name='dense_id_3')(id_2)
    id_2 = BatchNormalization()(id_2)
    id_2 = Activation('relu', name='activation_id_3')(id_2)
    id_2 = Dropout(0.5)(id_2)

    # merge attr_id embedding
    merge_attr_id_embedding = merge([attr_1, id_2], mode='concat',name='merge_attr_id')
    dense_1 = Dense(64,name='dense_merge_attr_id')(merge_attr_id_embedding)
    dense_1 = BatchNormalization()(dense_1)
    dense_1 = Activation('relu',name='activation')(dense_1)
    dense_1 = Dropout(0.5)(dense_1)
    # dense_1=BatchNormalization()(dense_1)
    #    dense_1=Dropout(0.2)(dense_1)

    # dense_2=Dense(16)(dense_1)
    # dense_2=Activation('relu')(dense_2)
    #    dense_2=BatchNormalization()(dense_2)
    #    dense_2=Dropout(0.2)(dense_2)

    # dense_3=Dense(8)(dense_2)
    # dense_3=Activation('relu')(dense_3)
    #    dense_3=BatchNormalization()(dense_3)
    #    dense_3=Dropout(0.2)(dense_3)

    # topLayer = Dense(1, activation='sigmoid', init='lecun_uniform',
    #                  name='topLayer')(dense_1)
    topLayer = Dense(1,  init='lecun_uniform',
                      name='topLayer')(dense_1)
    topLayer = BatchNormalization()(topLayer)
    topLayer = Activation(activation='sigmoid')(topLayer)
    # topLayer = Dropout(0.2)(topLayer)

    # Final prediction layer
    model = Model(input=[user_fea_input, item_fea_input, user_id_input, item_id_input],
                  output=topLayer)
    return model


def get_model11(num_users, num_items,max_seq_len):
    ########################   attr side   ##################################

    # Input
    user_fea_input = Input(shape=(max_seq_len,), dtype='float32', name='user_fea_input')
    # word_emb = Embedding(vocab_size, emb_size, name='word_emb')
    # user_attr_embedding=Flatten(name='flatten_user_fea')(word_emb(user_fea_input))#(?,200,100)
    user_attr_embedding = Dense(64, name='dense_user_fea')(user_fea_input)#(?,200,8)
    user_attr_embedding = BatchNormalization()(user_attr_embedding)
    user_attr_embedding = Activation(activation='relu')(user_attr_embedding)
    user_attr_embedding = Dropout(0.3)(user_attr_embedding)
    user_attr_embedding = Reshape((1,64),name='reshape_user_fea')(user_attr_embedding)

    item_fea_input = Input(shape=(max_seq_len,), dtype='float32', name='item_fea_input')
    # item_attr_embedding=Flatten(name='flatten_item_fea')(word_emb(item_fea_input))
    item_attr_embedding = Dense(64,name='dense_item_fea')(item_fea_input)
    item_attr_embedding = BatchNormalization()(item_attr_embedding)
    item_attr_embedding = Activation(activation='relu')(item_attr_embedding)
    item_attr_embedding = Dropout(0.3)(item_attr_embedding)
    item_attr_embedding = Reshape((64, 1),name='reshape_item_fea')(item_attr_embedding)

    # merge_attr_embedding=Multiply()(user_fea_input,item_fea_input)
    # merge_attr_embedding = merge([user_fea_input, item_fea_input], mode='mul')  # element-wise multiply

    merge_attr_embedding = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[1, 2]),name='lambda_merge')(
        [user_attr_embedding, item_attr_embedding])

    merge_attr_embedding_global = Flatten(name='flatten_merge_attr_global')(merge_attr_embedding)

    merge_attr_embedding = Reshape((64, 64, 1),name='reshape_merge_attr')(merge_attr_embedding)
#用一层卷积还是多层卷积需要尝试
    merge_attr_embedding = Conv2D(64, (3, 3),name='conv')(merge_attr_embedding)
    merge_attr_embedding = BatchNormalization(axis=3,name='normalization')(merge_attr_embedding)#归一化
    #这里加不加最大池化，dense层等需要尝试
    merge_attr_embedding = Activation('relu',name='activation_merge_attr')(merge_attr_embedding)#local_vector
    merge_attr_embedding = MaxPooling2D((2, 2))(merge_attr_embedding)
    merge_attr_embedding = Dropout(0.7)(merge_attr_embedding)

    merge_attr_embedding = Conv2D(32, (3, 3), name='conv1')(merge_attr_embedding)
    merge_attr_embedding = BatchNormalization(axis=3, name='normalization1')(merge_attr_embedding)  # 归一化
    # 这里加不加最大池化，dense层等需要尝试
    merge_attr_embedding = Activation('relu', name='activation_merge_attr1')(merge_attr_embedding)  # local_vector
    merge_attr_embedding = MaxPooling2D((2, 2))(merge_attr_embedding)
    merge_attr_embedding = Dropout(0.7)(merge_attr_embedding)

    # merge_attr_embedding = Conv2D(32, (3, 3))(merge_attr_embedding)
    # merge_attr_embedding = BatchNormalization(axis=3)(merge_attr_embedding)
    # merge_attr_embedding = Activation('relu')(merge_attr_embedding)
    # merge_attr_embedding = MaxPooling2D((2, 2))(merge_attr_embedding)

    # merge_attr_embedding = Conv2D(8, (3, 3))(merge_attr_embedding)
    # merge_attr_embedding = BatchNormalization(axis=3)(merge_attr_embedding)
    # merge_attr_embedding = Activation('relu')(merge_attr_embedding)

    merge_attr_embedding = Flatten(name='flatten_merge_attr')(merge_attr_embedding)
    merge_attr_embedding = merge([merge_attr_embedding, merge_attr_embedding_global], mode='concat',name='merge')

    attr_1 = Dense(32,name='dense_attr_1')(merge_attr_embedding)#16隐藏节点的个数需要调
    attr_1=BatchNormalization()(attr_1)#是否加归一化需要尝试
    attr_1 = Activation('relu',name='activation_attr1')(attr_1)
    attr_1=Dropout(0.7)(attr_1)

    # attr_2 = Dense(16)(attr_1)
    # attr_2 = Activation('relu')(attr_2)
    #    id_2=BatchNormalization()(id_2)
    #    id_2=Dropout(0.2)(id_2)

    ########################   id side   ##################################

    user_id_input = Input(shape=(1,), dtype='float32', name='user_id_input')
    user_id_Embedding = Embedding(input_dim=num_users, output_dim=32, name='user_id_Embedding',
                                  embeddings_initializer=RandomNormal(
                                      mean=0.0, stddev=0.01, seed=None),
                                  W_regularizer=l2(0), input_length=1)
    user_id_Embedding = Flatten(name='flatten_user_id')(user_id_Embedding(user_id_input))

    item_id_input = Input(shape=(1,), dtype='float32', name='item_id_input')
    item_id_Embedding = Embedding(input_dim=num_items, output_dim=32, name='item_id_Embedding',
                                  embeddings_initializer=RandomNormal(
                                      mean=0.0, stddev=0.01, seed=None),
                                  W_regularizer=l2(0), input_length=1)
    item_id_Embedding = Flatten(name='flatten_item_id')(item_id_Embedding(item_id_input))

    # id merge embedding
    merge_id_embedding = merge([user_id_Embedding, item_id_Embedding], mode='mul',name='merge_id')
    # id_1 = Dense(64)(merge_id_embedding)
    # id_1 = Activation('relu')(id_1)
    #MLP部分，几层还需要尝试
    id_2 = Dense(32,name='dense_id_2')(merge_id_embedding)
    id_2 = BatchNormalization()(id_2)
    id_2 = Activation('relu',name='activation_id_2')(id_2)
    id_2 = Dropout(0.5)(id_2)

    # merge attr_id embedding
    merge_attr_id_embedding = merge([attr_1, id_2], mode='concat',name='merge_attr_id')
    dense_1 = Dense(64,name='dense_merge_attr_id')(merge_attr_id_embedding)
    dense_1 = BatchNormalization()(dense_1)
    dense_1 = Activation('relu',name='activation')(dense_1)
    dense_1 = Dropout(0.5)(dense_1)
    # dense_1=BatchNormalization()(dense_1)
    #    dense_1=Dropout(0.2)(dense_1)

    # dense_2=Dense(16)(dense_1)
    # dense_2=Activation('relu')(dense_2)
    #    dense_2=BatchNormalization()(dense_2)
    #    dense_2=Dropout(0.2)(dense_2)

    # dense_3=Dense(8)(dense_2)
    # dense_3=Activation('relu')(dense_3)
    #    dense_3=BatchNormalization()(dense_3)
    #    dense_3=Dropout(0.2)(dense_3)

    # topLayer = Dense(1, activation='sigmoid', init='lecun_uniform',
    #                  name='topLayer')(dense_1)
    topLayer = Dense(1,  init='lecun_uniform',
                     name='topLayer')(dense_1)
    topLayer = BatchNormalization()(topLayer)
    topLayer = Activation(activation='sigmoid')(topLayer)
    # topLayer = Dropout(0.2)(topLayer)

    # Final prediction layer
    model = Model(input=[user_fea_input, item_fea_input, user_id_input, item_id_input],
                  output=topLayer)
    return model

def get_model12(num_users, num_items,max_seq_len):
    ########################   attr side   ##################################

    # Input
    user_fea_input = Input(shape=(max_seq_len,), dtype='float32', name='user_fea_input')
    # word_emb = Embedding(vocab_size, emb_size, name='word_emb')
    # user_attr_embedding=Flatten(name='flatten_user_fea')(word_emb(user_fea_input))#(?,200,100)
    user_attr_embedding = Dense(64, name='dense_user_fea')(user_fea_input)#(?,200,8)
    user_attr_embedding = BatchNormalization()(user_attr_embedding)
    user_attr_embedding = Activation(activation='relu')(user_attr_embedding)
    # user_attr_embedding = Dropout(0.3)(user_attr_embedding)
    user_attr_embedding = Reshape((1,64),name='reshape_user_fea')(user_attr_embedding)

    item_fea_input = Input(shape=(max_seq_len,), dtype='float32', name='item_fea_input')
    # item_attr_embedding=Flatten(name='flatten_item_fea')(word_emb(item_fea_input))
    item_attr_embedding = Dense(64,name='dense_item_fea')(item_fea_input)
    item_attr_embedding = BatchNormalization()(item_attr_embedding)
    item_attr_embedding = Activation(activation='relu')(item_attr_embedding)
    # item_attr_embedding = Dropout(0.3)(item_attr_embedding)
    item_attr_embedding = Reshape((64, 1),name='reshape_item_fea')(item_attr_embedding)

    # merge_attr_embedding=Multiply()(user_fea_input,item_fea_input)
    # merge_attr_embedding = merge([user_fea_input, item_fea_input], mode='mul')  # element-wise multiply

    merge_attr_embedding = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[1, 2]),name='lambda_merge')(
        [user_attr_embedding, item_attr_embedding])

    merge_attr_embedding_global = Flatten(name='flatten_merge_attr_global')(merge_attr_embedding)

    merge_attr_embedding = Reshape((64, 64, 1),name='reshape_merge_attr')(merge_attr_embedding)
#用一层卷积还是多层卷积需要尝试
    merge_attr_embedding = Conv2D(64, (3, 3),name='conv')(merge_attr_embedding)
    merge_attr_embedding = BatchNormalization(axis=3,name='normalization')(merge_attr_embedding)#归一化
    #这里加不加最大池化，dense层等需要尝试
    merge_attr_embedding = Activation('relu',name='activation_merge_attr')(merge_attr_embedding)#local_vector
    merge_attr_embedding = MaxPooling2D((2, 2))(merge_attr_embedding)
    merge_attr_embedding = Dropout(0.7)(merge_attr_embedding)

    merge_attr_embedding = Conv2D(32, (3, 3), name='conv1')(merge_attr_embedding)
    merge_attr_embedding = BatchNormalization(axis=3, name='normalization1')(merge_attr_embedding)  # 归一化
    # 这里加不加最大池化，dense层等需要尝试
    merge_attr_embedding = Activation('relu', name='activation_merge_attr1')(merge_attr_embedding)  # local_vector
    merge_attr_embedding = MaxPooling2D((2, 2))(merge_attr_embedding)
    merge_attr_embedding = Dropout(0.7)(merge_attr_embedding)

    # merge_attr_embedding = Conv2D(32, (3, 3))(merge_attr_embedding)
    # merge_attr_embedding = BatchNormalization(axis=3)(merge_attr_embedding)
    # merge_attr_embedding = Activation('relu')(merge_attr_embedding)
    # merge_attr_embedding = MaxPooling2D((2, 2))(merge_attr_embedding)

    # merge_attr_embedding = Conv2D(8, (3, 3))(merge_attr_embedding)
    # merge_attr_embedding = BatchNormalization(axis=3)(merge_attr_embedding)
    # merge_attr_embedding = Activation('relu')(merge_attr_embedding)

    merge_attr_embedding = Flatten(name='flatten_merge_attr')(merge_attr_embedding)
    merge_attr_embedding = merge([merge_attr_embedding, merge_attr_embedding_global], mode='concat',name='merge')

    attr_1 = Dense(128, name='dense_attr_1')(merge_attr_embedding)  # 16隐藏节点的个数需要调
    attr_1 = BatchNormalization()(attr_1)  # 是否加归一化需要尝试
    attr_1 = Activation('relu', name='activation_attr1')(attr_1)
    attr_1 = Dropout(0.7)(attr_1)
    attr_1 = Dense(64,name='dense_attr_2')(attr_1)#16隐藏节点的个数需要调
    attr_1=BatchNormalization()(attr_1)#是否加归一化需要尝试
    attr_1 = Activation('relu',name='activation_attr2')(attr_1)
    attr_1=Dropout(0.7)(attr_1)

    # attr_2 = Dense(16)(attr_1)
    # attr_2 = Activation('relu')(attr_2)
    #    id_2=BatchNormalization()(id_2)
    #    id_2=Dropout(0.2)(id_2)

    ########################   id side   ##################################

    user_id_input = Input(shape=(1,), dtype='float32', name='user_id_input')
    user_id_Embedding = Embedding(input_dim=num_users, output_dim=32, name='user_id_Embedding',
                                  embeddings_initializer=RandomNormal(
                                      mean=0.0, stddev=0.01, seed=None),
                                  W_regularizer=l2(0), input_length=1)
    user_id_Embedding = Flatten(name='flatten_user_id')(user_id_Embedding(user_id_input))

    item_id_input = Input(shape=(1,), dtype='float32', name='item_id_input')
    item_id_Embedding = Embedding(input_dim=num_items, output_dim=32, name='item_id_Embedding',
                                  embeddings_initializer=RandomNormal(
                                      mean=0.0, stddev=0.01, seed=None),
                                  W_regularizer=l2(0), input_length=1)
    item_id_Embedding = Flatten(name='flatten_item_id')(item_id_Embedding(item_id_input))

    # id merge embedding
    merge_id_embedding = merge([user_id_Embedding, item_id_Embedding], mode='mul',name='merge_id')
    # id_1 = Dense(64)(merge_id_embedding)
    # id_1 = Activation('relu')(id_1)
    #MLP部分，几层还需要尝试
    id_2 = Dense(16,name='dense_id_2')(merge_id_embedding)
    id_2 = BatchNormalization()(id_2)
    id_2 = Activation('relu',name='activation_id_2')(id_2)
    id_2 = Dropout(0.3)(id_2)

    # merge attr_id embedding
    merge_attr_id_embedding = merge([attr_1, id_2], mode='concat',name='merge_attr_id')
    dense_1 = Dense(64,name='dense_merge_attr_id')(merge_attr_id_embedding)
    dense_1 = BatchNormalization()(dense_1)
    dense_1 = Activation('relu',name='activation')(dense_1)
    dense_1 = Dropout(0.5)(dense_1)
    # dense_1=BatchNormalization()(dense_1)
    #    dense_1=Dropout(0.2)(dense_1)

    # dense_2=Dense(16)(dense_1)
    # dense_2=Activation('relu')(dense_2)
    #    dense_2=BatchNormalization()(dense_2)
    #    dense_2=Dropout(0.2)(dense_2)

    # dense_3=Dense(8)(dense_2)
    # dense_3=Activation('relu')(dense_3)
    #    dense_3=BatchNormalization()(dense_3)
    #    dense_3=Dropout(0.2)(dense_3)

    # topLayer = Dense(1, activation='sigmoid', init='lecun_uniform',
    #                  name='topLayer')(dense_1)
    topLayer = Dense(1,  init='lecun_uniform',
                     name='topLayer')(dense_1)
    topLayer = BatchNormalization()(topLayer)
    topLayer = Activation(activation='sigmoid')(topLayer)
    # topLayer = Dropout(0.2)(topLayer)

    # Final prediction layer
    model = Model(input=[user_fea_input, item_fea_input, user_id_input, item_id_input],
                  output=topLayer)
    return model


def get_model13(num_users, num_items,max_seq_len):
    ########################   attr side   ##################################

    # Input
    user_fea_input = Input(shape=(max_seq_len,), dtype='float32', name='user_fea_input')
    # word_emb = Embedding(vocab_size, emb_size, name='word_emb')
    # user_attr_embedding=Flatten(name='flatten_user_fea')(word_emb(user_fea_input))#(?,200,100)
    user_attr_embedding = Dense(64, name='dense_user_fea')(user_fea_input)#(?,200,8)
    user_attr_embedding = BatchNormalization()(user_attr_embedding)
    user_attr_embedding = Activation(activation='relu')(user_attr_embedding)
    # user_attr_embedding = Dropout(0.3)(user_attr_embedding)
    user_attr_embedding = Reshape((1,64),name='reshape_user_fea')(user_attr_embedding)

    item_fea_input = Input(shape=(max_seq_len,), dtype='float32', name='item_fea_input')
    # item_attr_embedding=Flatten(name='flatten_item_fea')(word_emb(item_fea_input))
    item_attr_embedding = Dense(64,name='dense_item_fea')(item_fea_input)
    item_attr_embedding = BatchNormalization()(item_attr_embedding)
    item_attr_embedding = Activation(activation='relu')(item_attr_embedding)
    # item_attr_embedding = Dropout(0.3)(item_attr_embedding)
    item_attr_embedding = Reshape((64, 1),name='reshape_item_fea')(item_attr_embedding)

    # merge_attr_embedding=Multiply()(user_fea_input,item_fea_input)
    # merge_attr_embedding = merge([user_fea_input, item_fea_input], mode='mul')  # element-wise multiply

    merge_attr_embedding = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[1, 2]),name='lambda_merge')(
        [user_attr_embedding, item_attr_embedding])

    merge_attr_embedding_global = Flatten(name='flatten_merge_attr_global')(merge_attr_embedding)

    merge_attr_embedding = Reshape((64, 64, 1),name='reshape_merge_attr')(merge_attr_embedding)
#用一层卷积还是多层卷积需要尝试
    merge_attr_embedding = Conv2D(64, (3, 3),name='conv')(merge_attr_embedding)
    merge_attr_embedding = BatchNormalization(axis=3,name='normalization')(merge_attr_embedding)#归一化
    #这里加不加最大池化，dense层等需要尝试
    merge_attr_embedding = Activation('relu',name='activation_merge_attr')(merge_attr_embedding)#local_vector
    # merge_attr_embedding = MaxPooling2D((2, 2))(merge_attr_embedding)
    # merge_attr_embedding = Dropout(0.7)(merge_attr_embedding)

    # merge_attr_embedding = Conv2D(32, (3, 3), name='conv1')(merge_attr_embedding)
    # merge_attr_embedding = BatchNormalization(axis=3, name='normalization1')(merge_attr_embedding)  # 归一化
    # # 这里加不加最大池化，dense层等需要尝试
    # merge_attr_embedding = Activation('relu', name='activation_merge_attr1')(merge_attr_embedding)  # local_vector
    # merge_attr_embedding = MaxPooling2D((2, 2))(merge_attr_embedding)
    # merge_attr_embedding = Dropout(0.7)(merge_attr_embedding)

    # merge_attr_embedding = Conv2D(32, (3, 3))(merge_attr_embedding)
    # merge_attr_embedding = BatchNormalization(axis=3)(merge_attr_embedding)
    # merge_attr_embedding = Activation('relu')(merge_attr_embedding)
    # merge_attr_embedding = MaxPooling2D((2, 2))(merge_attr_embedding)

    # merge_attr_embedding = Conv2D(8, (3, 3))(merge_attr_embedding)
    # merge_attr_embedding = BatchNormalization(axis=3)(merge_attr_embedding)
    # merge_attr_embedding = Activation('relu')(merge_attr_embedding)

    merge_attr_embedding = Flatten(name='flatten_merge_attr')(merge_attr_embedding)
    merge_attr_embedding = merge([merge_attr_embedding, merge_attr_embedding_global], mode='concat',name='merge')

    attr_1 = Dense(64,name='dense_attr_1')(merge_attr_embedding)#16隐藏节点的个数需要调
    attr_1=BatchNormalization()(attr_1)#是否加归一化需要尝试
    attr_1 = Activation('relu',name='activation_attr1')(attr_1)
    # attr_1=Dropout(0.7)(attr_1)

    # attr_2 = Dense(16)(attr_1)
    # attr_2 = Activation('relu')(attr_2)
    #    id_2=BatchNormalization()(id_2)
    #    id_2=Dropout(0.2)(id_2)

    ########################   id side   ##################################

    user_id_input = Input(shape=(1,), dtype='float32', name='user_id_input')
    user_id_Embedding = Embedding(input_dim=num_users, output_dim=32, name='user_id_Embedding',
                                  embeddings_initializer=RandomNormal(
                                      mean=0.0, stddev=0.01, seed=None),
                                  W_regularizer=l2(0), input_length=1)
    user_id_Embedding = Flatten(name='flatten_user_id')(user_id_Embedding(user_id_input))

    item_id_input = Input(shape=(1,), dtype='float32', name='item_id_input')
    item_id_Embedding = Embedding(input_dim=num_items, output_dim=32, name='item_id_Embedding',
                                  embeddings_initializer=RandomNormal(
                                      mean=0.0, stddev=0.01, seed=None),
                                  W_regularizer=l2(0), input_length=1)
    item_id_Embedding = Flatten(name='flatten_item_id')(item_id_Embedding(item_id_input))

    # id merge embedding
    merge_id_embedding = merge([user_id_Embedding, item_id_Embedding], mode='mul',name='merge_id')
    # id_1 = Dense(64)(merge_id_embedding)
    # id_1 = Activation('relu')(id_1)
    #MLP部分，几层还需要尝试
    id_2 = Dense(32,name='dense_id_2')(merge_id_embedding)
    id_2 = BatchNormalization()(id_2)
    id_2 = Activation('relu',name='activation_id_2')(id_2)
    # id_2 = Dropout(0.5)(id_2)

    # merge attr_id embedding
    merge_attr_id_embedding = merge([attr_1, id_2], mode='concat',name='merge_attr_id')
    dense_1 = Dense(64,name='dense_merge_attr_id')(merge_attr_id_embedding)
    dense_1 = BatchNormalization()(dense_1)
    dense_1 = Activation('relu',name='activation')(dense_1)
    # dense_1 = Dropout(0.5)(dense_1)
    # dense_1=BatchNormalization()(dense_1)
    #    dense_1=Dropout(0.2)(dense_1)

    # dense_2=Dense(16)(dense_1)
    # dense_2=Activation('relu')(dense_2)
    #    dense_2=BatchNormalization()(dense_2)
    #    dense_2=Dropout(0.2)(dense_2)

    # dense_3=Dense(8)(dense_2)
    # dense_3=Activation('relu')(dense_3)
    #    dense_3=BatchNormalization()(dense_3)
    #    dense_3=Dropout(0.2)(dense_3)

    # topLayer = Dense(1, activation='sigmoid', init='lecun_uniform',
    #                  name='topLayer')(dense_1)
    topLayer = Dense(1,  init='lecun_uniform',
                     name='topLayer')(dense_1)
    topLayer = BatchNormalization()(topLayer)
    topLayer = Activation(activation='sigmoid')(topLayer)
    # topLayer = Dropout(0.2)(topLayer)

    # Final prediction layer
    model = Model(input=[user_fea_input, item_fea_input, user_id_input, item_id_input],
                  output=topLayer)
    return model
def get_model14(num_users, num_items):
    ########################   attr side   ##################################

    # Input
    user_fea_input = Input(shape=(max_seq_len,), dtype='float32', name='user_fea_input')
    # word_emb = Embedding(vocab_size, emb_size, name='word_emb')
    # user_attr_embedding=Flatten(name='flatten_user_fea')(word_emb(user_fea_input))#(?,200,100)
    user_attr_embedding = Dense(64, name='dense_user_fea')(user_fea_input)  # (?,200,8)
    user_attr_embedding = BatchNormalization()(user_attr_embedding)
    user_attr_embedding = Activation(activation='relu')(user_attr_embedding)
    # user_attr_embedding = Dropout(0.3)(user_attr_embedding)
    user_attr_embedding = Reshape((1, 64), name='reshape_user_fea')(user_attr_embedding)

    item_fea_input = Input(shape=(max_seq_len,), dtype='float32', name='item_fea_input')
    # item_attr_embedding=Flatten(name='flatten_item_fea')(word_emb(item_fea_input))
    item_attr_embedding = Dense(64, name='dense_item_fea')(item_fea_input)
    item_attr_embedding = BatchNormalization()(item_attr_embedding)
    item_attr_embedding = Activation(activation='relu')(item_attr_embedding)
    # item_attr_embedding = Dropout(0.3)(item_attr_embedding)
    item_attr_embedding = Reshape((64, 1), name='reshape_item_fea')(item_attr_embedding)

    # merge_attr_embedding=Multiply()(user_fea_input,item_fea_input)
    # merge_attr_embedding = merge([user_fea_input, item_fea_input], mode='mul')  # element-wise multiply

    merge_attr_embedding = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[1, 2]), name='lambda_merge')(
        [user_attr_embedding, item_attr_embedding])

    merge_attr_embedding_global = Flatten(name='flatten_merge_attr_global')(merge_attr_embedding)
    merge_attr_embedding_global = Dense(64)(merge_attr_embedding_global)
    merge_attr_embedding_global = BatchNormalization()(merge_attr_embedding_global)
    merge_attr_embedding_global = Activation('relu')(merge_attr_embedding_global)

    # merge_attr_embedding = Reshape((64, 64, 1), name='reshape_merge_attr')(merge_attr_embedding)
    # # 用一层卷积还是多层卷积需要尝试
    # merge_attr_embedding = Conv2D(64, (3, 3), name='conv')(merge_attr_embedding)
    # merge_attr_embedding = BatchNormalization(axis=3, name='normalization')(merge_attr_embedding)  # 归一化
    # # 这里加不加最大池化，dense层等需要尝试
    # merge_attr_embedding = Activation('relu', name='activation_merge_attr')(merge_attr_embedding)  # local_vector
    # # merge_attr_embedding = MaxPooling2D((2, 2))(merge_attr_embedding)
    # # merge_attr_embedding = Dropout(0.7)(merge_attr_embedding)
    #
    # # merge_attr_embedding = Conv2D(32, (3, 3), name='conv1')(merge_attr_embedding)
    # # merge_attr_embedding = BatchNormalization(axis=3, name='normalization1')(merge_attr_embedding)  # 归一化
    # # # 这里加不加最大池化，dense层等需要尝试
    # # merge_attr_embedding = Activation('relu', name='activation_merge_attr1')(merge_attr_embedding)  # local_vector
    # # merge_attr_embedding = MaxPooling2D((2, 2))(merge_attr_embedding)
    # # merge_attr_embedding = Dropout(0.7)(merge_attr_embedding)
    #
    # # merge_attr_embedding = Conv2D(32, (3, 3))(merge_attr_embedding)
    # # merge_attr_embedding = BatchNormalization(axis=3)(merge_attr_embedding)
    # # merge_attr_embedding = Activation('relu')(merge_attr_embedding)
    # # merge_attr_embedding = MaxPooling2D((2, 2))(merge_attr_embedding)
    #
    # # merge_attr_embedding = Conv2D(8, (3, 3))(merge_attr_embedding)
    # # merge_attr_embedding = BatchNormalization(axis=3)(merge_attr_embedding)
    # # merge_attr_embedding = Activation('relu')(merge_attr_embedding)
    #
    # merge_attr_embedding = Flatten(name='flatten_merge_attr')(merge_attr_embedding)
    # merge_attr_embedding = merge([merge_attr_embedding, merge_attr_embedding_global], mode='concat', name='merge')
    #
    # attr_1 = Dense(128, name='dense_attr_1')(merge_attr_embedding)  # 16隐藏节点的个数需要调
    # attr_1 = BatchNormalization()(attr_1)  # 是否加归一化需要尝试
    # attr_1 = Activation('relu', name='activation_attr1')(attr_1)
    # attr_1 = Dense(64, name='dense_attr_2')(attr_1)  # 16隐藏节点的个数需要调
    # attr_1 = BatchNormalization()(attr_1)  # 是否加归一化需要尝试
    # attr_1 = Activation('relu', name='activation_attr2')(attr_1)
    # # attr_1=Dropout(0.7)(attr_1)
    #
    # # attr_2 = Dense(16)(attr_1)
    # # attr_2 = Activation('relu')(attr_2)
    # #    id_2=BatchNormalization()(id_2)
    # #    id_2=Dropout(0.2)(id_2)


    topLayer = Dense(1, init='lecun_uniform',
                     name='topLayer')(merge_attr_embedding_global)
    topLayer = BatchNormalization()(topLayer)
    topLayer = Activation(activation='sigmoid')(topLayer)
    # topLayer = Dropout(0.2)(topLayer)

    # Final prediction layer
    model = Model(input=[user_fea_input, item_fea_input],
                  output=topLayer)
    return model

def get_model15(num_users, num_items,max_seq_len):
    ########################   attr side   ##################################

    # Input
    user_fea_input = Input(shape=(max_seq_len,), dtype='float32', name='user_fea_input')
    # word_emb = Embedding(vocab_size, emb_size, name='word_emb')
    # user_attr_embedding=Flatten(name='flatten_user_fea')(word_emb(user_fea_input))#(?,200,100)
    user_attr_embedding = Dense(64, name='dense_user_fea')(user_fea_input)  # (?,200,8)
    user_attr_embedding = BatchNormalization()(user_attr_embedding)
    user_attr_embedding = Activation(activation='relu')(user_attr_embedding)
    # user_attr_embedding = Dropout(0.3)(user_attr_embedding)
    user_attr_embedding = Reshape((1, 64), name='reshape_user_fea')(user_attr_embedding)

    item_fea_input = Input(shape=(max_seq_len,), dtype='float32', name='item_fea_input')
    # item_attr_embedding=Flatten(name='flatten_item_fea')(word_emb(item_fea_input))
    item_attr_embedding = Dense(64, name='dense_item_fea')(item_fea_input)
    item_attr_embedding = BatchNormalization()(item_attr_embedding)
    item_attr_embedding = Activation(activation='relu')(item_attr_embedding)
    # item_attr_embedding = Dropout(0.3)(item_attr_embedding)
    item_attr_embedding = Reshape((64, 1), name='reshape_item_fea')(item_attr_embedding)

    # merge_attr_embedding=Multiply()(user_fea_input,item_fea_input)
    # merge_attr_embedding = merge([user_fea_input, item_fea_input], mode='mul')  # element-wise multiply

    merge_attr_embedding = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[1, 2]), name='lambda_merge')(
        [user_attr_embedding, item_attr_embedding])

    merge_attr_embedding_global = Flatten(name='flatten_merge_attr_global')(merge_attr_embedding)

    merge_attr_embedding = Reshape((64, 64, 1), name='reshape_merge_attr')(merge_attr_embedding)
    # 用一层卷积还是多层卷积需要尝试
    merge_attr_embedding = Conv2D(64, (3, 3), name='conv')(merge_attr_embedding)
    merge_attr_embedding = BatchNormalization(axis=3, name='normalization')(merge_attr_embedding)  # 归一化
    # 这里加不加最大池化，dense层等需要尝试
    merge_attr_embedding = Activation('relu', name='activation_merge_attr')(merge_attr_embedding)  # local_vector
    # merge_attr_embedding = MaxPooling2D((2, 2))(merge_attr_embedding)
    # merge_attr_embedding = Dropout(0.7)(merge_attr_embedding)

    # merge_attr_embedding = Conv2D(32, (3, 3), name='conv1')(merge_attr_embedding)
    # merge_attr_embedding = BatchNormalization(axis=3, name='normalization1')(merge_attr_embedding)  # 归一化
    # # 这里加不加最大池化，dense层等需要尝试
    # merge_attr_embedding = Activation('relu', name='activation_merge_attr1')(merge_attr_embedding)  # local_vector
    # merge_attr_embedding = MaxPooling2D((2, 2))(merge_attr_embedding)
    # merge_attr_embedding = Dropout(0.7)(merge_attr_embedding)

    # merge_attr_embedding = Conv2D(32, (3, 3))(merge_attr_embedding)
    # merge_attr_embedding = BatchNormalization(axis=3)(merge_attr_embedding)
    # merge_attr_embedding = Activation('relu')(merge_attr_embedding)
    # merge_attr_embedding = MaxPooling2D((2, 2))(merge_attr_embedding)

    # merge_attr_embedding = Conv2D(8, (3, 3))(merge_attr_embedding)
    # merge_attr_embedding = BatchNormalization(axis=3)(merge_attr_embedding)
    # merge_attr_embedding = Activation('relu')(merge_attr_embedding)

    merge_attr_embedding = Flatten(name='flatten_merge_attr')(merge_attr_embedding)
    merge_attr_embedding = merge([merge_attr_embedding, merge_attr_embedding_global], mode='concat', name='merge')

    attr_1 = Dense(128, name='dense_attr_1')(merge_attr_embedding)  # 16隐藏节点的个数需要调
    attr_1 = BatchNormalization()(attr_1)  # 是否加归一化需要尝试
    attr_1 = Activation('relu', name='activation_attr1')(attr_1)
    attr_1 = Dense(64, name='dense_attr_2')(attr_1)  # 16隐藏节点的个数需要调
    attr_1 = BatchNormalization()(attr_1)  # 是否加归一化需要尝试
    attr_1 = Activation('relu', name='activation_attr2')(attr_1)

    attr_1 = Dense(32, name='dense_attr_3')(attr_1)  # 16隐藏节点的个数需要调
    attr_1 = BatchNormalization()(attr_1)  # 是否加归一化需要尝试
    attr_1 = Activation('relu', name='activation_attr3')(attr_1)
    # attr_1=Dropout(0.7)(attr_1)

    # attr_2 = Dense(16)(attr_1)
    # attr_2 = Activation('relu')(attr_2)
    #    id_2=BatchNormalization()(id_2)
    #    id_2=Dropout(0.2)(id_2)

    ########################   id side   ##################################

    user_id_input = Input(shape=(1,), dtype='float32', name='user_id_input')
    user_id_Embedding = Embedding(input_dim=num_users, output_dim=64, name='user_id_Embedding',
                                  embeddings_initializer=RandomNormal(
                                      mean=0.0, stddev=0.01, seed=None),
                                  W_regularizer=l2(0), input_length=1)
    user_id_Embedding = Flatten(name='flatten_user_id')(user_id_Embedding(user_id_input))

    item_id_input = Input(shape=(1,), dtype='float32', name='item_id_input')
    item_id_Embedding = Embedding(input_dim=num_items, output_dim=64, name='item_id_Embedding',
                                  embeddings_initializer=RandomNormal(
                                      mean=0.0, stddev=0.01, seed=None),
                                  W_regularizer=l2(0), input_length=1)
    item_id_Embedding = Flatten(name='flatten_item_id')(item_id_Embedding(item_id_input))

    # id merge embedding
    merge_id_embedding = merge([user_id_Embedding, item_id_Embedding], mode='mul', name='merge_id')
    # id_1 = Dense(64)(merge_id_embedding)
    # id_1 = Activation('relu')(id_1)
    # MLP部分，几层还需要尝试

    id_2 = Dense(64, name='dense_id_2_1')(merge_id_embedding)
    id_2 = BatchNormalization()(id_2)
    id_2 = Activation('relu', name='activation_id_2_1')(id_2)
    id_2 = Dense(32, name='dense_id_2')(merge_id_embedding)
    id_2 = BatchNormalization()(id_2)
    id_2 = Activation('relu', name='activation_id_2')(id_2)
    # id_2 = Dropout(0.5)(id_2)

    # merge attr_id embedding
    merge_attr_id_embedding = merge([attr_1, id_2], mode='concat', name='merge_attr_id')
    dense_1 = Dense(64, name='dense_merge_attr_id')(merge_attr_id_embedding)
    dense_1 = BatchNormalization()(dense_1)
    dense_1 = Activation('relu', name='activation')(dense_1)

    dense_1 = Dense(32, name='dense_merge_attr_id1')(dense_1)
    dense_1 = BatchNormalization()(dense_1)
    dense_1 = Activation('relu', name='activation1')(dense_1)
    # dense_1 = Dropout(0.5)(dense_1)
    # dense_1=BatchNormalization()(dense_1)
    #    dense_1=Dropout(0.2)(dense_1)

    # dense_2=Dense(16)(dense_1)
    # dense_2=Activation('relu')(dense_2)
    #    dense_2=BatchNormalization()(dense_2)
    #    dense_2=Dropout(0.2)(dense_2)

    # dense_3=Dense(8)(dense_2)
    # dense_3=Activation('relu')(dense_3)
    #    dense_3=BatchNormalization()(dense_3)
    #    dense_3=Dropout(0.2)(dense_3)

    # topLayer = Dense(1, activation='sigmoid', init='lecun_uniform',
    #                  name='topLayer')(dense_1)
    topLayer = Dense(1, init='lecun_uniform',
                     name='topLayer')(dense_1)
    topLayer = BatchNormalization()(topLayer)
    topLayer = Activation(activation='sigmoid')(topLayer)
    # topLayer = Dropout(0.2)(topLayer)

    # Final prediction layer
    model = Model(input=[user_fea_input, item_fea_input, user_id_input, item_id_input],
                  output=topLayer)
    return model
def main():
    learning_rate = 0.00001
    num_epochs = 50
    verbose = 1
    topK = 10
    evaluation_threads = 1
    num_negatives = 4
    startTime = time()
    # emb_size=100#每个单词的词向量的维度
    # emb_path='../preprocessed_data/w2v_embedding'
    # max_seq_len=100
    # vocab_size = 1156158

    # load data
    # num_users, users_attr_mat = load_user_attributes()
    # num_items, items_genres_mat = load_itemGenres_as_matrix()
    # max_seq_len=load_max_seq_len()#每一条记录的文本的最大长度
    print("load user_review")
    user_review_fea=load_review_feature('../data_50_word/yelp_user_fea_vector.csv')#一个字典
    print("load item_review")
    item_review_fea=load_review_feature('../data_50_word/yelp_item_fea_vector.csv')
    #     # users_vec_mat = oad_user_vectors()
    print("load train")
    train = load_rating_file_as_matrix('../data_50_word/yelp_train_7.csv')
    num_users, num_items = train.shape
    # print("load vocab")
    # vocab=create_vocab('../data_50_word/Amazon_test_data.csv')

    # load model
    model = get_model14(num_users, num_items)

    # compile model
    model.compile(
        optimizer=Adam(lr=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', 'mae']
    )
    # plot_model(model, show_shapes=True, to_file='mainMovieUserCnn.png')
    model.summary()

    # Training model
    best_hr, best_ndcg = 0, 0
    data_x=[]
    y_loss=[]
    y_val_loss=[]
    y_HR=[]
    y_NDCG=[]
    for epoch in range(num_epochs):
        data_x.append(epoch)
        print('The %d epoch...............................' % (epoch))
        t1 = time()
        # Generate training instances
        user_fea_input,item_fea_input, labels = get_train_instances(train,user_review_fea,item_review_fea)

        hist = model.fit([user_fea_input, item_fea_input],
                         labels, epochs=1,
                         batch_size=256,
                         verbose=2,
                         shuffle=True,
                         validation_split=0.1)
        t2 = time()

        # y_loss.append(hist.history['loss'][0])
        # y_val_loss.append(hist.history['val_loss'][0])

        # Evaluation
        if epoch % verbose == 0:
            testRatings = load_rating_file_as_list('../data_50_word/yelp_test_7.csv')
            testNegatives = load_negative_file('../data_50_word/yelp_test_negative_7.csv')
            (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives,
                                           topK, evaluation_threads)
            hr, ndcg, loss,val_loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0],hist.history['val_loss'][0]
            y_loss.append(loss)
            y_val_loss.append(val_loss)
            y_HR.append(hr)
            y_NDCG.append(ndcg)
            print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f,val_loss = %.4f [%.1f s]'
                  % (epoch, t2 - t1, hr, ndcg, loss,val_loss, time() - t2))
            if hr > best_hr:
                best_hr = hr
                if hr > 0.7:
                    print("save weights")
                    model.save_weights('../data_50_word/Amazon_global_neg_%d_hr_%.4f_ndcg_%.4f.h5' %
                                       (num_negatives, hr, ndcg), overwrite=True)
            if ndcg > best_ndcg:
                best_ndcg = ndcg
    endTime = time()
    print("End. best HR = %.6f, best NDCG = %.4f,time = %.1f s" %
          (best_hr, best_ndcg, endTime - startTime))
    print('HR = %.4f, NDCG = %.4f' % (hr, ndcg))
    plt.plot(data_x,y_loss,label='loss')
    plt.plot(data_x,y_val_loss,label='val_loss')
    plt.plot(data_x,y_HR,label='HR')
    plt.plot(data_x,y_NDCG,label='NDCG')
    plt.legend()
    plt.savefig('../doc2vec/model13.png')
    plt.show()
    print("y_loss",y_loss)
    print("y_val_loss",y_val_loss)
    print("HR",y_HR)
    print("NDCG",y_NDCG)


if __name__ == '__main__':
    main()
