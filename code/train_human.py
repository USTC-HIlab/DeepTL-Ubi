#!/usr/bin/env python3

import functools
import itertools
import os
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.layers import Dense, Activation, Flatten, Dropout, Reshape
from keras.layers import Input, Lambda
from keras.layers import Conv1D,Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from keras import optimizers
from keras.optimizers import Adam,SGD
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.models import Model
from keras import backend as K
import my_densenet

filefolder = 'uniprot seq/data/'
species1 = 'HUMAN'
species2 = 'HUMAN'
window_size = 31
phase = 'train'
domain_adaptation_task = 'HUMAN_HUMAN'

def save_feature(feature,phase,index):
    l = feature.shape[1]
    num = feature.shape[0]
    final_matrix = np.ones((num,l))
    for i in range(num):
        temp = feature[i].flatten()
        final_matrix[i] = temp
    if not os.path.exists(feafolder):
        os.makedirs(feafolder)
    np.savetxt(feafolder + '/feature_{:s}_{:d}.txt'.format(phase,index),final_matrix)
    
def Create_Pairs(domain_adaptation_task,phase,species1,species2,window_size):

    UM  = domain_adaptation_task   
    print 'Creating pairs for repetition: '
    
    X_train_target=np.load('uniprot seq/data_npy/{:s}-{:s}-{:d}_feature.npy'.format(phase,species1,window_size))
    y_train_target=np.load('uniprot seq/data_npy/{:s}-{:s}-{:d}_label.npy'.format(phase,species1,window_size))

    X_train_source=np.load('uniprot seq/data_npy/{:s}-{:s}-{:d}_feature.npy'.format(phase,species2,window_size))
    y_train_source=np.load('uniprot seq/data_npy/{:s}-{:s}-{:d}_label.npy'.format(phase,species2,window_size))
    print(y_train_source[0])
    print(y_train_target[0])
    
    Training_P=[]
    Training_N=[]

    l_target = len(y_train_target)
    list_l_target = range(0,l_target)
    random.shuffle(list_l_target)
    trt = 0
    for trs in range(len(y_train_source)):
        if y_train_source[trs]==y_train_target[list_l_target[trt]]:
            Training_P.append([trs,list_l_target[trt]])
        else:
            Training_N.append([trs,list_l_target[trt]])
        trt = trt + 1
        if trt>=l_target:
            trt = 0
            
    Training = Training_P+Training_N
    random.shuffle(Training)


    X1=np.zeros([len(Training),31,21],dtype='float32')
    X2=np.zeros([len(Training),31,21],dtype='float32')

    y1=np.zeros([len(Training)])
    y2=np.zeros([len(Training)])
    yc=np.zeros([len(Training)])

    for i in range(len(Training)):
        in1,in2=Training[i]
        X1[i,:,:]=X_train_source[in1,:,:]
        X2[i,:,:]=X_train_target[in2,:,:]

        y1[i]=y_train_source[in1]
        y2[i]=y_train_target[in2]
        if y_train_source[in1]==y_train_target[in2]:
            yc[i]=1

    if not os.path.exists('pairs'):
        os.makedirs('pairs')

    np.save('CCSA-master/pairs/' + UM + '_X1_count.npy', X1)
    np.save('CCSA-master/pairs/' + UM + '_X2_count.npy', X2)

    np.save('CCSA-master/pairs/' + UM + '_y1_count.npy', y1)
    np.save('CCSA-master/pairs/' + UM + '_y2_count.npy', y2)
    np.save('CCSA-master/pairs/' + UM + '_yc_count.npy', yc)

# Create_Pairs(domain_adaptation_task,phase,species1,species2,window_size)

def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def training_the_model(domain_adaptation_task,species1,species2,window_size):
    UM = domain_adaptation_task
    mainfolder = '/harddisk/hdd_d/liuyu/PTM/ubi/new-data/domain/CCSA-master'
    phase = 'valid'
    X_valid = np.load('uniprot seq/data_npy/{:s}-{:s}-{:d}_feature.npy'.format(phase,species1,window_size))
    y_valid = np.load('uniprot seq/data_npy/{:s}-{:s}-{:d}_label.npy'.format(phase,species1,window_size))
    X_valid = X_valid.reshape(X_valid.shape[0], 31, 21)
    y_valid = np_utils.to_categorical(y_valid, nb_classes)


    X_valid_s = np.load('uniprot seq/data_npy/{:s}-{:s}-{:d}_feature.npy'.format(phase,species2,window_size))
    y_valid_s= np.load('uniprot seq/data_npy/{:s}-{:s}-{:d}_label.npy'.format(phase,species2,window_size))
    X_valid_s = X_valid_s.reshape(X_valid_s.shape[0], 31, 21)
    y_valid_s = np_utils.to_categorical(y_valid_s, nb_classes)
    
    phase = 'test'
    X_test = np.load('uniprot seq/data_npy/{:s}-{:s}-{:d}_feature.npy'.format(phase,species1,window_size))
    y_test= np.load('uniprot seq/data_npy/{:s}-{:s}-{:d}_label.npy'.format(phase,species1,window_size))
    X_test = X_test.reshape(X_test.shape[0], 31, 21)
    y_test = np_utils.to_categorical(y_test, nb_classes)
    
    X1 = np.load(mainfolder + '/pairs/' + UM + '_X1_count.npy')
    X2 = np.load(mainfolder +'/pairs/' + UM + '_X2_count.npy')

    X1 = X1.reshape(X1.shape[0], 31, 21)
    X2 = X2.reshape(X2.shape[0], 31, 21)

    y1 = np.load(mainfolder +'/pairs/'+ UM + '_y1_count.npy')
    y2 = np.load(mainfolder +'/pairs/'+ UM + '_y2_count.npy')
    yc = np.load(mainfolder +'/pairs/'+ UM + '_yc_count.npy')

    y1 = np_utils.to_categorical(y1, nb_classes)
    y2 = np_utils.to_categorical(y2, nb_classes)
    
    img_dim = X1.shape[1:]
    opt = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model = my_densenet.DenseNet(nb_classes, nb_layers,img_dim, init_form, nb_dense_block,
             growth_rate,filter_size_block,
             nb_filter, filter_size_ori,
             dense_number,dropout_rate,dropout_dense,weight_decay
             )
#     model.compile(loss={'classification': 'binary_crossentropy', 'CSA':contrastive_loss},
#               optimizer='adadelta',
#               loss_weights={'classification': 0.8, 'CSA': 0.2})
    
    model.compile(loss={'classification': 'binary_crossentropy', 'CSA':contrastive_loss},
              optimizer= opt,
              loss_weights={'classification': weigth, 'CSA': 1-weigth})

#     model.load_weights('model_dense80.h5')
    print 'Training the model - Epoch '+str(nb_epoch)
    nn=batch_size
    best_Acc = 0
    best_Auc = 0
    best_Auc_epoch = 0
    best_Acc_epoch = 0
    
    loss_train = []
    loss_train_h = []
    loss_valid = []
    loss_test = []
    acc_valid = []
    acc_train = []
    acc_train_h = []
    acc_test = []
    index_fine = 0
    
    mainfolder = 'CCSA-master/result/' + species1
    if not os.path.exists(mainfolder):
                os.makedirs(mainfolder)
            
    for e in range(nb_epoch):
#         if e % 10 == 0:
#             print(str(e) + '->')
#         for i in range(len(y2) / nn):
#             loss1 = model.train_on_batch([X1[i * nn:(i + 1) * nn, :, :], X2[i * nn:(i + 1) * nn, :, :]],
#                                         [y1[i * nn:(i + 1) * nn, :], yc[i * nn:(i + 1) * nn, ]])
#         loss_train_h.append(loss1)
#         np.savetxt(mainfolder + '/train_loss_h.txt',loss_train_h)
#         Out = model.predict([X_valid_s, X_valid_s]) 
#         Acc_v = np.argmax(Out[0], axis=1) - np.argmax(y_valid_s, axis=1)
#         Acc = (len(Acc_v) - np.count_nonzero(Acc_v) + .0000001) / len(Acc_v)
#         auc = metrics.roc_auc_score( y_valid_s[:, 1],Out[0][:, 1])
#         print('HUMAN AUC in epoch {:d} is {:f}'.format(e,auc))
#         print('HUMAN ACC in epoch {:d} is {:f}'.format(e,Acc))
#         print('loss1 is', loss1)  
        
#         model.save_weights(modelfolder+'/model_dense' + str(e+1) + '.h5', overwrite=True)
        if e % 1 == 0:
            index_fine = index_fine+1
            for i in range(len(y2) / nn):
                loss2 = model.train_on_batch([X2[i * nn:(i + 1) * nn, :, :], X1[i * nn:(i + 1) * nn, :, :]],
                                        [y2[i * nn:(i + 1) * nn, :], yc[i * nn:(i + 1) * nn, ]])
            loss_train.append(loss2)
            np.savetxt(mainfolder+'/train_loss.txt',loss_train)    
            if not os.path.exists(modelfolder+ '/fine'):
                os.makedirs(modelfolder+ '/fine')
            model.save_weights(modelfolder+ '/fine'+'/model_dense' + str(e+1) + '.h5', overwrite=True)
            Out = model.predict([X_valid, X_valid])
            valid_result = model.evaluate([X_valid,X_valid], [y_valid,y_valid])
            loss_valid.append(valid_result)
            np.savetxt(mainfolder+'/valid_loss.txt',loss_valid)
            Acc_v = np.argmax(Out[0], axis=1) - np.argmax(y_valid, axis=1)
            Acc = (len(Acc_v) - np.count_nonzero(Acc_v) + .0000001) / len(Acc_v)
            print('loss2 is', loss2)
            auc = metrics.roc_auc_score( y_valid[:, 1],Out[0][:, 1])
            
            
            Out = model.predict([X_test, X_test])
            np.savetxt(prefolder+ '/prediction scores of epoch-{:d}.txt'.format(e+1),Out[0])
            test_result = model.evaluate([X_test,X_test], [y_test,y_test])
            loss_test.append(test_result)
            np.savetxt(mainfolder+'/test_loss.txt',loss_test)
            
            dense1_layer_model = Model(inputs=model.input,outputs=model.get_layer('dense1').output)
            dense1_output = dense1_layer_model.predict([X_test,X_test], batch_size=None) 
            print('zhongjianceng')
            save_feature(dense1_output,'independent',e)
            fig, ax = plt.subplots()
            x = range(0, index_fine)
            loss_train_arr = np.array(loss_train)
            loss_valid_arr = np.array(loss_valid)
            loss_test_arr = np.array(loss_test)
            plt.plot(x, loss_train_arr[:,0], 'r-', label='train loss ')
            ax.legend(loc='upper right', shadow=True)
            plt.plot(x, loss_valid_arr[:,0], 'b-', label='validation loss ')
            ax.legend(loc='upper right', shadow=True)
            plt.plot(x, loss_test_arr[:,0], 'g-', label='test loss ')
            ax.legend(loc='upper right', shadow=True)
            plt.title('loss of densenet {:s} epoch-{:d}'.format(species1, nb_epoch))
            plt.xlabel('epoch')
            plt.savefig(mainfolder+'/loss of epoch {:d}.png'.format(nb_epoch))
    
    
            Acc_v = np.argmax(Out[0], axis=1) - np.argmax(y_test, axis=1)
            Acc = (len(Acc_v) - np.count_nonzero(Acc_v) + .0000001) / len(Acc_v)
            auc = metrics.roc_auc_score(y_test[:, 1],Out[0][:, 1])
            
            print('Target AUC in epoch {:d} is {:f}'.format(e,auc))
            print('Target ACC in epoch {:d} is {:f}'.format(e,Acc))
            if best_Acc < Acc:
                best_Acc = Acc
                best_Acc_epoch = e
            print('best Acc in epcho is',best_Acc_epoch,best_Acc)
            if best_Auc < auc:
                best_Auc = auc
                best_Auc_epoch = e
            print('best Auc in epoch is',best_Auc_epoch,best_Auc)
            
            fpr_t, tpr_t, _ = metrics.roc_curve(y_test[:, 1],Out[0][:, 1])
            fig, ax = plt.subplots()
            ax.plot(fpr_t, tpr_t, 'b-', label='CNN-test {:.3%}'.format(auc))
            ax.legend(loc='lower right', shadow=True)
            plt.title('test of {:s}'.format(species1))
            plt.savefig(figfolder+'/ptm-{:s}-epoch-{:d}-auc-{:.4f}-acc-{:.4f}-test.png'.format(species1, e+1, auc, Acc))
            plt.close()
            
            print('Target test AUC in epoch {:d} is {:f}'.format(e,auc))
            print('Target test ACC in epoch {:d} is {:f}'.format(e,Acc))
        print('best Acc in epcho is',best_Acc_epoch,best_Acc) 
        print('best Auc in epoch is',best_Auc_epoch,best_Auc)
    print str(e)
#     return best_Acc

  
if __name__ == '__main__':
    Create_Pairs(domain_adaptation_task,phase,species1,species2,window_size) 
    mainfolder = 'CCSA-master/result/' + species1
    if not os.path.exists(mainfolder):
                os.makedirs(mainfolder)
    figfolder = mainfolder + '/figure'
    if not os.path.exists(figfolder):
                os.makedirs(figfolder)
    prefolder = mainfolder + '/prediction'
    if not os.path.exists(prefolder):
                os.makedirs(prefolder)
    feafolder = mainfolder + '/feature'
    if not os.path.exists(feafolder):
                os.makedirs(feafolder)
    modelfolder = mainfolder + '/model'
    if not os.path.exists(modelfolder):
                os.makedirs(modelfolder)
    os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'  # '1' TITAN 1080
    weigth = 1.0
    nb_epoch = 200
    window_size = 31
    nb_classes = 2
    batch_size = 256
    init_form = 'RandomUniform'
    learning_rate = 0.0002
    opt_model = 'adam'
    nb_dense_block = 1
    nb_layers = 6
    nb_filter = 32
    growth_rate = 16
    filter_size_block = 7
    filter_size_ori = 1
    dense_number = 64
    dropout_rate = 0.2
    dropout_dense = 0.3
    weight_decay = 0.0001
    training_the_model(domain_adaptation_task,species1,species2,window_size)
