'''
Created on Sept 24, 2018

Keras Implementation of A^3NCF rating prediction model in:
CHEGN Zhiyong et al. A^3NCF: An Adaptive Aspect Attention Model for Rating Prediction, In IJCAI 2018.
@author Zhiyong Cheng (jason.zy.cheng@gmail.com)

The code was developed based on Dr. Xiangnan He's NCF codes (https://github.com/hexiangnan/neural_collaborative_filtering).
@author: Xiangnan He (xiangnanhe@gmail.com)
'''
import gc
import time
from time import time
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.initializers import RandomNormal
from keras.layers import Dense, Activation, Flatten, Lambda, Reshape, MaxPooling2D, AveragePooling2D,MaxPooling1D,add
from keras.layers import Embedding, Input, merge, Conv2D,Multiply,Dropout,Concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
import numpy as np
import keras
from keras import backend as K
from keras import initializers
from keras.models import Sequential, Model, load_model, save_model
# from keras.layers.core import Dense, Lambda, Activation
# from keras.layers import Embedding, Input, Dense, merge, Reshape,Conv1D, MaxPooling1D, Flatten,Conv2D
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from .DataSet import load_rating_file_as_matrix
from .DataSet import load_review_feature
import matplotlib.pyplot as plt
from .lCoupledCF_evaluate import eval_mae_rmse
from time import time
import multiprocessing as mp
import sys
import math
import argparse


# KERAS_BACKEND=tensorflow python -c "from keras import backend"
#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run GMF.")
    parser.add_argument('--path', nargs='?', default='../datasets/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='Amazon',
                        help='Choose a dataset.')
    parser.add_argument('--k', type=int, default=5,
                        help='Number of latent topics in represnetation')
    parser.add_argument('--activation_function', nargs='?', default='hard_sigmoid',
                        help='activation functions')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--num_factors', type=int, default=5,
                        help='Embedding size.')
    parser.add_argument('--regs', nargs='?', default='[0,0]',
                        help="Regularization for user and item embeddings.")
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
    parser.add_argument('--emb_size', type=int, default=100,
                        help='CNN embedding size')
    parser.add_argument('--filters', type=int, default=2,
                        help='CNN filters.')
    parser.add_argument('--kernel_size', type=int, default=10,
                        help='CNN kernel_size.')
    parser.add_argument('--hidden_size', type=int, default=64,
                        help='CNN hidden_size.')
    # parser.add_argument('--u_seq_size', type=int, default=10,
    #                     help='CNN user features length.')
    # parser.add_argument('--i_seq_size', type=int, default=10,
    #                     help='CNN item features length.')

    return parser.parse_args()


def init_normal(shape, name=None):
    return K.random_normal(shape, mean=0, stddev=0.01, seed=None)


from keras.utils.generic_utils import get_custom_objects


def clipped_relu(x):
    return K.relu(x, max_value=1)


def vallina_relu(x):
    # return inputs * K.cast(K.greater(inputs, self.theta), K.floatx())
    # noise = K.random_normal((8,1), mean=0, stddev=0.00001)
    return K.cast(K.greater(x, 0), K.floatx()) + 0.0001


def get_model(num_users, num_items,max_seq_len):
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

    dense_1=Dense(128,activation='relu')(merge_attr_embedding_global)
    dense_1 = Dense(64,activation='relu')(dense_1)
    topLayer = Dense(1,activation='relu',
                     name='topLayer')(dense_1)


    # Final prediction layer
    model = Model(input=[user_fea_input, item_fea_input],
                  output=topLayer)
    return model


def get_model1(num_users, num_items,max_seq_len):
    ########################   attr side   ##################################

    # Input
    user_fea_input = Input(shape=(max_seq_len,), dtype='float32', name='user_fea_input')
    # word_emb = Embedding(vocab_size, emb_size, name='word_emb')
    # user_attr_embedding=Flatten(name='flatten_user_fea')(word_emb(user_fea_input))#(?,200,100)
    user_attr_embedding = Dense(32, name='dense_user_fea')(user_fea_input)  # (?,200,8)
    user_attr_embedding = BatchNormalization()(user_attr_embedding)
    user_attr_embedding = Activation(activation='relu')(user_attr_embedding)
    # user_attr_embedding = Dropout(0.3)(user_attr_embedding)
    user_attr_embedding = Reshape((1, 32), name='reshape_user_fea')(user_attr_embedding)

    item_fea_input = Input(shape=(max_seq_len,), dtype='float32', name='item_fea_input')
    # item_attr_embedding=Flatten(name='flatten_item_fea')(word_emb(item_fea_input))
    item_attr_embedding = Dense(32, name='dense_item_fea')(item_fea_input)
    item_attr_embedding = BatchNormalization()(item_attr_embedding)
    item_attr_embedding = Activation(activation='relu')(item_attr_embedding)
    # item_attr_embedding = Dropout(0.3)(item_attr_embedding)
    item_attr_embedding = Reshape((32, 1), name='reshape_item_fea')(item_attr_embedding)

    # merge_attr_embedding=Multiply()(user_fea_input,item_fea_input)
    # merge_attr_embedding = merge([user_fea_input, item_fea_input], mode='mul')  # element-wise multiply

    merge_attr_embedding = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[1, 2]), name='lambda_merge')(
        [user_attr_embedding, item_attr_embedding])

    merge_attr_embedding_global = Flatten(name='flatten_merge_attr_global')(merge_attr_embedding)


    topLayer = Dense(64,activation='relu')(merge_attr_embedding_global)
    topLayer = Dense(32,activation='relu')(merge_attr_embedding_global)
    topLayer = Dense(1, name='topLayer',activation='relu')(topLayer)

    # topLayer = Dropout(0.2)(topLayer)

    # Final prediction layer
    model = Model(input=[user_fea_input,item_fea_input],
                  output=topLayer)
    return model

def get_model2(num_users, num_items,max_seq_len):
    ########################   attr side   ##################################

    # Input
    user_fea_input = Input(shape=(max_seq_len,), dtype='float32', name='user_fea_input')
    # word_emb = Embedding(vocab_size, emb_size, name='word_emb')
    # user_attr_embedding=Flatten(name='flatten_user_fea')(word_emb(user_fea_input))#(?,200,100)
    user_attr_embedding = Dense(32, name='dense_user_fea',activation='relu')(user_fea_input)  # (?,200,8)
    # user_attr_embedding = BatchNormalization()(user_attr_embedding)
    # user_attr_embedding = Activation(activation='relu')(user_attr_embedding)
    # user_attr_embedding = Dropout(0.3)(user_attr_embedding)
    user_attr_embedding = Reshape((1, 32), name='reshape_user_fea')(user_attr_embedding)

    item_fea_input = Input(shape=(max_seq_len,), dtype='float32', name='item_fea_input')
    # item_attr_embedding=Flatten(name='flatten_item_fea')(word_emb(item_fea_input))
    item_attr_embedding = Dense(32, name='dense_item_fea',activation='relu')(item_fea_input)
    # item_attr_embedding = BatchNormalization()(item_attr_embedding)
    # item_attr_embedding = Activation(activation='relu')(item_attr_embedding)
    # item_attr_embedding = Dropout(0.3)(item_attr_embedding)
    item_attr_embedding = Reshape((32, 1), name='reshape_item_fea')(item_attr_embedding)

    # merge_attr_embedding=Multiply()(user_fea_input,item_fea_input)
    # merge_attr_embedding = merge([user_fea_input, item_fea_input], mode='mul')  # element-wise multiply

    merge_attr_embedding = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[1, 2]), name='lambda_merge')(
        [user_attr_embedding, item_attr_embedding])

    merge_attr_embedding_global = Flatten(name='flatten_merge_attr_global')(merge_attr_embedding)

    topLayer = Dense(64,activation='relu')(merge_attr_embedding_global)
    topLayer = Dense(32,activation='relu')(topLayer)
    topLayer = Dense(1, name='topLayer',activation='relu')(topLayer)

    # topLayer = Dropout(0.2)(topLayer)

    # Final prediction layer
    model = Model(input=[user_fea_input,item_fea_input],
                  output=topLayer)
    return model

def get_model3(num_users, num_items,max_seq_len):
    ########################   attr side   ##################################

    # Input
    user_fea_input = Input(shape=(max_seq_len,), dtype='float32', name='user_fea_input')
    # word_emb = Embedding(vocab_size, emb_size, name='word_emb')
    # user_attr_embedding=Flatten(name='flatten_user_fea')(word_emb(user_fea_input))#(?,200,100)
    user_attr_embedding = Dense(64, name='dense_user_fea',activation='relu')(user_fea_input)  # (?,200,8)
    # user_attr_embedding = BatchNormalization()(user_attr_embedding)
    # user_attr_embedding = Activation(activation='relu')(user_attr_embedding)
    # user_attr_embedding = Dropout(0.3)(user_attr_embedding)
    user_attr_embedding = Reshape((1, 64), name='reshape_user_fea')(user_attr_embedding)

    item_fea_input = Input(shape=(max_seq_len,), dtype='float32', name='item_fea_input')
    # item_attr_embedding=Flatten(name='flatten_item_fea')(word_emb(item_fea_input))
    item_attr_embedding = Dense(64, name='dense_item_fea',activation='relu')(item_fea_input)
    # item_attr_embedding = BatchNormalization()(item_attr_embedding)
    # item_attr_embedding = Activation(activation='relu')(item_attr_embedding)
    # item_attr_embedding = Dropout(0.3)(item_attr_embedding)
    item_attr_embedding = Reshape((64, 1), name='reshape_item_fea')(item_attr_embedding)

    # merge_attr_embedding=Multiply()(user_fea_input,item_fea_input)
    # merge_attr_embedding = merge([user_fea_input, item_fea_input], mode='mul')  # element-wise multiply

    merge_attr_embedding = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[1, 2]), name='lambda_merge')(
        [user_attr_embedding, item_attr_embedding])

    merge_attr_embedding_global = Flatten(name='flatten_merge_attr_global')(merge_attr_embedding)

    topLayer = Dense(64,activation='relu')(merge_attr_embedding_global)
    topLayer = Dense(32,activation='relu')(topLayer)
    topLayer = Dense(1, name='topLayer',activation='relu')(topLayer)


    # Final prediction layer
    model = Model(input=[user_fea_input,item_fea_input],
                  output=topLayer)
    return model

def get_train_instances(train, user_review_fea,item_review_fea):
    user_fea, item_fea, labels = [], [], []
    num_users = train.shape[0]
    for (u, i) in train.keys():

        user_fea.append(user_review_fea[u])

        item_fea.append(item_review_fea[i])
        label = train[u, i]
        labels.append(label)
    return  np.array(user_fea, dtype='float32'),\
           np.array(item_fea, dtype='float32'), np.array(labels)

def main():
    learning_rate=0.00001
    epochs=100
    verbose=1
    max_seq_len=100
    vocab_size = 380606
    emb_size = 100
    args = parse_args()
    num_factors = 32
    k = max_seq_len
    regs = eval(args.regs)
    learner = args.learner
    # learning_rate = args.lr
    # epochs = args.epochs
    verbose = args.verbose
    activation_function = args.activation_function


    evaluation_threads = 1  # mp.cpu_count()
    # print("A3NCF arguments: %s" % (args))
    # model_out_file = 'Pretrain/%sNumofTopic_%d_GMF_%d_%d.h5' %(args.dataset, k, num_factors, time())

    # Loading data
    t1 = time()
    # dataset = Dataset(args.path + args.dataset, k)
    # train=load_rating_file_as_matrix('../data//Amazon_train.csv')
    train = load_rating_file_as_matrix('../data//yelp_train_7.csv')
    # user_review_fea=load_review_feature('../data_50_word/Amazon_user_fea_vector.csv')
    user_review_fea = load_review_feature('../data_50_word/yelp_user_fea_vector.csv')
    # item_review_fea=load_review_feature('../data_50_word/Amazon_item_fea_vector.csv')
    item_review_fea = load_review_feature('../data_50_word/yelp_item_fea_vector.csv')
    # user_review_fea = load_review_feature('../word2vec/Amazon_user_fea_vector.csv')
    # item_review_fea = load_review_feature('../word2vec/Amazon_item_fea_vector.csv')
    # testRatings = load_rating_file_as_matrix('../data//Amazon_test.csv')
    testRatings = load_rating_file_as_matrix('../data//yelp_test_7.csv')
    num_users, num_items = train.shape
    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d"
          % (time() - t1, num_users, num_items, train.nnz, len(testRatings)))

    # Build model
    # model = get_model(num_users, num_items,max_seq_len)
    model = get_model2(num_users, num_items, max_seq_len)

    model.compile(
        optimizer=Adam(lr=learning_rate),
        loss='mean_squared_error'
    )
    # print(model.summary())
    model.summary()
    # Init performance
    t1 = time()
    (mae, rmse) = eval_mae_rmse(model, testRatings, user_review_fea, item_review_fea)
    print('Init: MAE = %.4f, RMSE = %.4f\t [%.1f s]' % (mae, rmse, time() - t1))

    # Train model
    best_mae, best_rmse, best_iter = mae, rmse, -1
    x=[]
    y_loss=[]
    y_val_loss=[]
    for epoch in range(epochs):
        x.append(epoch)
        t1 = time()
        # Generate training instances
        user_fea,item_fea, labels = \
            get_train_instances(train, user_review_fea,item_review_fea)

        # user_fea,item_fea=user_review_fea,item_review_fea

        # Training
        hist = model.fit([ user_fea,item_fea],  # input
                         labels,  # labels
                         batch_size=64, epochs=1, verbose=2, shuffle=True,validation_split=0.1)
        t2 = time()

        # Evaluation
        if epoch % verbose == 0:
            (mae, rmse) = eval_mae_rmse(model, testRatings, user_review_fea, item_review_fea)
            loss = hist.history['loss'][0]
            y_loss.append(loss)
            y_val_loss.append(hist.history['val_loss'][0])
            print('Iteration %d [%.1f s]: mae = %.4f, rmse = %.4f, loss = %.4f [%.1f s]'
                  % (epoch, t2 - t1, mae, rmse, loss, time() - t2))
            if rmse < best_rmse:
                best_mae, best_rmse, best_iter = mae, rmse, epoch
                # if args.out > 0:
                #    model.save_weights(model_out_file, overwrite=True)

    print("End. Best Iteration %d:  mae = %.4f, rmse = %.4f. " % (best_iter, best_mae, best_rmse))
    plt.plot(x, y_loss, label='loss')
    plt.plot(x, y_val_loss, label='val_loss')
    plt.legend()
    plt.show()
    # if args.out > 0:
    #    print("The best ancf model is saved to %s" %(model_out_file))

    # outFile = 'results/ancf' + '.result'
    # f = open(outFile, 'a')
    # f.write(args.dataset + '\t' + activation_function + "\t" + str(num_factors) + '\t' + str(best_mae) + '\t' + str(
    #     best_rmse) + '\n')
    # f.close()
if __name__ == '__main__':
    main()
