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
from .LoadData import load_rating_file_as_matrix
from .LoadData import load_review_feature
from .LoadData import load_rating_file_as_list
from .LoadData import load_negative_file
from .evaluate import evaluate_model

def get_train_instances(ratings,user_review_fea, item_review_fea):
    '''
    :param ratings: 评分矩阵
    :param user_review_fea: 用户评论文本向量
    :param item_review_fea: 项目评论文本向量
    :return: 构造的模型输入特征
    '''
    user_id_input, user_fea,item_id_input,item_fea, labels = [], [], [],[],[]
    num_users, num_items = ratings.shape
    num_items=num_items-1
    num_negatives = 4
    for (u, i) in ratings.keys():
        # positive instance
        user_id_input.append([u])
        user_fea.append(user_review_fea[u])
        item_id_input.append([i])
        item_fea.append(item_review_fea[i])
        labels.append([1])
        # negative instances
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            while (u, j) in ratings:#过滤掉用户交互过的项目
                j = np.random.randint(num_items)
            user_id_input.append([u])
            user_fea.append(user_review_fea[u])
            item_id_input.append([j])
            item_fea.append(item_review_fea[j])
            labels.append([0])
    array_user_fea_input = np.array(user_fea)
    array_user_id_input = np.array(user_id_input)
    array_item_id_input = np.array(item_id_input)
    array_item_fea_input = np.array(item_fea)
    array_labels = np.array(labels)
    del user_fea, user_id_input, item_id_input, item_fea, labels
    gc.collect()
    return array_user_fea_input, array_user_id_input, array_item_fea_input, array_item_id_input, array_labels

def get_model(num_users, num_items,max_seq_len):
    ########################   attr side   ##################################
    # 用户评论文本处理
    user_fea_input = Input(shape=(max_seq_len,), dtype='float32', name='user_fea_input')
    user_attr_embedding = Dense(64, name='dense_user_fea')(user_fea_input)  # (?,200,8)
    user_attr_embedding = BatchNormalization()(user_attr_embedding)
    user_attr_embedding = Activation(activation='relu')(user_attr_embedding)
    user_attr_embedding = Reshape((1, 64), name='reshape_user_fea')(user_attr_embedding)
    #项目评论文本处理
    item_fea_input = Input(shape=(max_seq_len,), dtype='float32', name='item_fea_input')
    item_attr_embedding = Dense(64, name='dense_item_fea')(item_fea_input)
    item_attr_embedding = BatchNormalization()(item_attr_embedding)
    item_attr_embedding = Activation(activation='relu')(item_attr_embedding)
    item_attr_embedding = Reshape((64, 1), name='reshape_item_fea')(item_attr_embedding)
    #构造耦合矩阵
    merge_attr_embedding = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[1, 2]), name='lambda_merge')(
        [user_attr_embedding, item_attr_embedding])
    merge_attr_embedding = Reshape((64, 64, 1), name='reshape_merge_attr')(merge_attr_embedding)
    # 用一层卷积还是多层卷积需要尝试
    merge_attr_embedding = Conv2D(64, (3, 3), name='conv')(merge_attr_embedding)
    merge_attr_embedding = BatchNormalization(axis=3, name='normalization')(merge_attr_embedding)  # 归一化
    # 这里加不加最大池化，dense层等需要尝试
    merge_attr_embedding = Activation('relu', name='activation_merge_attr')(merge_attr_embedding)
    # local_vector
    merge_attr_embedding = Flatten(name='flatten_merge_attr')(merge_attr_embedding)
    #local vector
    # 将耦合矩阵压缩为一维向量
    merge_attr_embedding_global = Flatten(name='flatten_merge_attr_global')(merge_attr_embedding)
    #将局部向量和全局向量合并表示用户-项目耦合向量
    merge_attr_embedding = merge([merge_attr_embedding, merge_attr_embedding_global], mode='concat', name='merge')
    #送入全连接层
    attr_1 = Dense(128, name='dense_attr_1')(merge_attr_embedding)  # 16隐藏节点的个数需要调
    attr_1 = BatchNormalization()(attr_1)  # 是否加归一化需要尝试
    attr_1 = Activation('relu', name='activation_attr1')(attr_1)
    attr_1 = Dense(64, name='dense_attr_2')(attr_1)  # 16隐藏节点的个数需要调
    attr_1 = BatchNormalization()(attr_1)  # 是否加归一化需要尝试
    attr_1 = Activation('relu', name='activation_attr2')(attr_1)

    #DeepCF模型
    #输入用户、项目ID
    user_id_input = Input(shape=(1,), dtype='float32', name='user_id_input')
    #Embedding操作
    user_id_Embedding = Embedding(input_dim=num_users, output_dim=32, name='user_id_Embedding',
                                  embeddings_initializer=RandomNormal(
                                      mean=0.0, stddev=0.01, seed=None),
                                  W_regularizer=l2(0), input_length=1)
    user_id_Embedding = Flatten(name='flatten_user_id')(user_id_Embedding(user_id_input))
    #项目ID处理
    item_id_input = Input(shape=(1,), dtype='float32', name='item_id_input')
    item_id_Embedding = Embedding(input_dim=num_items, output_dim=32, name='item_id_Embedding',
                                  embeddings_initializer=RandomNormal(
                                      mean=0.0, stddev=0.01, seed=None),
                                  W_regularizer=l2(0), input_length=1)
    item_id_Embedding = Flatten(name='flatten_item_id')(item_id_Embedding(item_id_input))

    # 用户、项目id merge embedding
    merge_id_embedding = merge([user_id_Embedding, item_id_Embedding], mode='mul', name='merge_id')
    # MLP部分，几层还需要尝试
    id_2 = Dense(32, name='dense_id_2')(merge_id_embedding)
    id_2 = BatchNormalization()(id_2)
    id_2 = Activation('relu', name='activation_id_2')(id_2)
    # merge attr_id embedding，将局部/全局耦合向量和DeepCF合并
    merge_attr_id_embedding = merge([attr_1, id_2], mode='concat', name='merge_attr_id')
    #送入MLP
    dense_1 = Dense(64, name='dense_merge_attr_id')(merge_attr_id_embedding)
    dense_1 = BatchNormalization()(dense_1)
    dense_1 = Activation('relu', name='activation')(dense_1)
    dense_1 = Dense(32, name='dense_merge_attr_id1')(dense_1)
    dense_1 = BatchNormalization()(dense_1)
    dense_1 = Activation('relu', name='activation1')(dense_1)

    topLayer = Dense(1, init='lecun_uniform',
                     name='topLayer')(dense_1)
    topLayer = BatchNormalization()(topLayer)
    topLayer = Activation(activation='sigmoid')(topLayer)
    # Final prediction layer
    model = Model(input=[user_fea_input, item_fea_input, user_id_input, item_id_input],
                  output=topLayer)
    return model

def main():
    '''
    主函数
    :return:
    '''
    #参数初始化
    learning_rate = 0.00001#学习率
    num_epochs = 50#迭代次数
    verbose = 1
    topK = 10#推荐列表长度
    evaluation_threads = 1
    num_negatives = 4#负样本个数
    startTime = time()
    max_seq_len=100#评论文本向量最大长度
    #加载用户、项目评论文本
    print("load user_review")
    user_review_fea=load_review_feature('../data_50_word/Amazon_user_fea_vector.csv')#一个字典
    print("load item_review")
    item_review_fea=load_review_feature('../data_50_word/Amazon_item_fea_vector.csv')
    # 加载用户-项目评分矩阵
    print("load train")
    train = load_rating_file_as_matrix('../data_50_word/Amazon_train.csv')
    num_users, num_items = train.shape
    # load model
    model = get_model(num_users, num_items,max_seq_len)
    # compile model
    model.compile(
        optimizer=Adam(lr=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy','rmse', 'mae']
    )
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
        user_fea,user_id_input,item_fea,item_id_input, labels = get_train_instances(train,user_review_fea,item_review_fea)
        hist = model.fit([user_fea, item_fea, user_id_input, item_id_input],
                         labels, epochs=1,
                         batch_size=256,
                         verbose=2,
                         shuffle=True,
                         validation_split=0.1)
        t2 = time()
        y_loss.append(hist.history['loss'][0])
        y_val_loss.append(hist.history['val_loss'][0])

        # Evaluation
        if epoch % verbose == 0:
            testRatings = load_rating_file_as_list('../data_50_word/Amazon_test.csv')
            testNegatives = load_negative_file('../data_50_word/Amazon_test_negative.csv')
            (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives,
                                           user_review_fea, item_review_fea, topK, evaluation_threads)
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
