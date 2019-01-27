from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import AveragePooling2D, AveragePooling1D
from keras.layers.pooling import GlobalAveragePooling2D,GlobalAveragePooling1D
from keras.layers import Input, merge, Flatten, Lambda
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import keras.backend as K
from keras.layers import Conv1D,Conv2D, MaxPooling2D


def conv_factory(x, init_form, nb_filter, filter_size_block, dropout_rate=None, weight_decay=1E-4):
    """Apply BatchNorm, Relu 3x3Conv2D, optional dropout

    :param x: Input keras network
    :param nb_filter: int -- number of filters
    :param dropout_rate: int -- dropout rate
    :param weight_decay: int -- weight decay factor

    :returns: keras network with b_norm, relu and convolution2d added
    :rtype: keras network
    """
    #x = Activation('relu')(x)
    x = Conv1D(nb_filter, filter_size_block,
                      init=init_form,
                      activation='relu',
                      border_mode='same',
                      bias=False,
                      W_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def transition(x, init_form, nb_filter, dropout_rate=None, weight_decay=1E-4):
    """Apply BatchNorm, Relu 1x1Conv2D, optional dropout and Maxpooling2D

    :param x: keras model
    :param nb_filter: int -- number of filters
    :param dropout_rate: int -- dropout rate
    :param weight_decay: int -- weight decay factor

    :returns: model
    :rtype: keras model, after applying batch_norm, relu-conv, dropout, maxpool

    """
    x = BatchNormalization(axis=1,
                           gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    
    x = Conv1D(nb_filter, 1,
                      init=init_form,
                      activation='relu',
                      border_mode='same',
                      bias=False,
                      W_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    #x = AveragePooling2D((2, 2),padding='same')(x)
#     x = AveragePooling1D(pool_size=2, padding='same')(x)

    return x


def denseblock(x, init_form, nb_layers, nb_filter, growth_rate,filter_size_block,
               dropout_rate=None, weight_decay=1E-4):
    """Build a denseblock where the output of each
       conv_factory is fed to subsequent ones

    :param x: keras model
    :param nb_layers: int -- the number of layers of conv_
                      factory to append to the model.
    :param nb_filter: int -- number of filters
    :param dropout_rate: int -- dropout rate
    :param weight_decay: int -- weight decay factor

    :returns: keras model with nb_layers of conv_factory appended
    :rtype: keras model

    """

    list_feat = [x]
    concat_axis = -1

    for i in range(nb_layers):
        x = conv_factory(x, init_form, growth_rate, filter_size_block, dropout_rate, weight_decay)
        list_feat.append(x)
        x = merge(list_feat, mode='concat', concat_axis=concat_axis)
        nb_filter += growth_rate
    return x

def euclidean_distance(vects):
    eps = 1e-08
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), eps))



def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def DenseNet_temp(nb_classes, nb_layers,img_dim, init_form, nb_dense_block,
             growth_rate,filter_size_block,
             nb_filter, filter_size_ori,
             dense_number,dropout_rate,dropout_dense,weight_decay,model_input_1,des_name):
    """ Build the DenseNet model

    :param nb_classes: int -- number of classes
    :param img_dim: tuple -- (channels, rows, columns)
    :param depth: int -- how many layers
    :param nb_dense_block: int -- number of dense blocks to add to end
    :param growth_rate: int -- number of filters to add
    :param nb_filter: int -- number of filters
    :param dropout_rate: float -- dropout rate
    :param weight_decay: float -- weight decay
    :param nb_layers:int --numbers of layers in a dense block
    :param filter_size_ori: int -- filter size of first conv1d
    :param dropout_dense: float---drop out rate of dense

    :returns: keras model with nb_layers of conv_factory appended
    :rtype: keras model

    """
    # Initial convolution
    x_1 = Conv1D(nb_filter, filter_size_ori,
                      init = init_form,
                      activation='relu',
                      border_mode='same',
                      bias=False,
                      W_regularizer=l2(weight_decay))(model_input_1)
    
#     x_2 = Conv1D(nb_filter, filter_size_ori,
#                       init = init_form,
#                       activation='relu',
#                       border_mode='same',
#                       name='initial_conv1D',
#                       bias=False,
#                       W_regularizer=l2(weight_decay))(model_input_2)


    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        x_1 = denseblock(x_1, init_form, nb_layers, nb_filter, growth_rate,filter_size_block,
                                  dropout_rate=dropout_rate,
                                  weight_decay=weight_decay)
        # add transition
        x_1 = transition(x_1, init_form, nb_filter, dropout_rate=dropout_rate,
                       weight_decay=weight_decay)
    
#     for block_idx in range(nb_dense_block - 1):
#         x_2 = denseblock(x_2, init_form, nb_layers, nb_filter, growth_rate,filter_size_block,
#                                   dropout_rate=dropout_rate,
#                                   weight_decay=weight_decay)
#         # add transition
#         x_2 = transition(x_2, init_form, nb_filter, dropout_rate=dropout_rate,
#                        weight_decay=weight_decay)
        
    # The last denseblock does not have a transition
    x_1 = denseblock(x_1, init_form, nb_layers, nb_filter, growth_rate,filter_size_block,
                              dropout_rate=dropout_rate,
                              weight_decay=weight_decay)
    
#     x_2 = denseblock(x_2, init_form, nb_layers, nb_filter, growth_rate,filter_size_block,
#                               dropout_rate=dropout_rate,
#                               weight_decay=weight_decay)
    
    x_1 = Activation('relu')(x_1)
    
#     x_2  = Activation('relu')(x_2 )

    #x = GlobalAveragePooling1D()(x)

    x_1 = Flatten()(x_1)
#     x_2 = Flatten()(x_2)

    x_1 = Dense(dense_number,
              activation='relu',init = init_form,
              W_regularizer=l2(weight_decay),
              b_regularizer=l2(weight_decay),name = des_name)(x_1)
    
#     x_2_out = Dense(dense_number,
#               activation='relu',init = init_form,
#               W_regularizer=l2(weight_decay),
#               b_regularizer=l2(weight_decay),name = 'dense2')(x_2)

    return x_1

def DenseNet(nb_classes, nb_layers,img_dim, init_form, nb_dense_block,
             growth_rate,filter_size_block,
             nb_filter, filter_size_ori,
             dense_number,dropout_rate,dropout_dense,weight_decay):
    model_input_1 = Input(shape=img_dim)
    model_input_2 = Input(shape=img_dim)
    x_1_out = DenseNet_temp(nb_classes, nb_layers,img_dim, init_form, nb_dense_block,
             growth_rate,filter_size_block,
             nb_filter, filter_size_ori,
             dense_number,dropout_rate,dropout_dense,weight_decay,model_input_1,'dense1')
    x_2= DenseNet_temp(nb_classes, nb_layers,img_dim, init_form, nb_dense_block,
             growth_rate,filter_size_block,
             nb_filter, filter_size_ori,
             dense_number,dropout_rate,dropout_dense,weight_decay,model_input_2,'dense2')
    
    x_1 = Dropout(dropout_dense)(x_1_out)
    #softmax
    x_1 = Dense(nb_classes,
              activation='softmax',init = init_form,
              W_regularizer=l2(weight_decay),
              b_regularizer=l2(weight_decay), name= 'classification')(x_1)
    
    distance = Lambda(euclidean_distance, output_shape = eucl_dist_output_shape, name='CSA')(
        [x_1_out, x_2])
    
    densenet = Model(input=[model_input_1,model_input_2], outputs=[x_1, x_2, x_1_out,distance],name="DenseNet")
    return densenet