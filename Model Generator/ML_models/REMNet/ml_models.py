# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 09:23:24 2022

@author: turtle9
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn import svm
# from ML_classes.CNN_class import convolutional_neural_network_mitigation
# from ML_classes.SVM_class import support_vector_machine
# from ML_classes.REMNet_multi_output_class import REMNet_multi_output
# from UWB_mitigation_helper_functions.definitions import correction_methods, cases

def cnn_mitigation_model(input_length):
    ## based on "Deep Learning Methodologies for UWB Ranging Error Compensation"
    model = models.Sequential()
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(16, 10, activation='relu', padding="same", input_shape=((input_length, 1, 1))))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(16, 10, activation='relu', padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(16, 10, activation='relu', padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32, 10, activation='relu', padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, 10, activation='relu', padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, 10, activation='relu', padding="same"))    
    model.add(layers.BatchNormalization())
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation="linear"))
    return model


def se_module(x, filters, ratio=8):
    
    avg_pool = tf.keras.layers.GlobalAveragePooling2D()(x)    
    

    avg_pool = tf.keras.layers.Dense(filters//ratio,
                             activation='relu')(avg_pool)

    excitation = tf.keras.layers.Dense(filters, activation='sigmoid')(avg_pool)
    excitation = tf.keras.layers.Reshape((1,filters))(excitation)
    
    return tf.keras.layers.Multiply()([x, excitation])

def red_module(x, filters, kernel_size):
    x_res = x
    
    x = tf.keras.layers.Conv2D(filters, kernel_size, strides=2, padding='same', activation='relu')(x)
    x_res = tf.keras.layers.Conv2D(filters, 1, strides=2, padding='same', activation='relu')(x_res)
    
    return tf.keras.layers.Add()([x_res, x])

def resa_red_module(x, filters,  kernel_size):
    x_res = x
    
    x = tf.keras.layers.Conv2D(filters, kernel_size, strides=1, padding='same', activation='relu')(x)
    x = se_module(x, filters, ratio=8)
    x = tf.keras.layers.Add()([x, x_res])
    return red_module(x, filters, kernel_size)

def resa_module(x, filters,  kernel_size):
    x_res = x
    
    x = tf.keras.layers.Conv2D(filters, kernel_size, strides=1, padding='same', activation='relu')(x)
    x = tf.keras.layers.Add()([x, x_res])
    return x

def REMNet_mitigation_original_model(input_shape, filters = 16, kernel_size = (3,1), n_modules = 3):
    input_tensor = tf.keras.layers.Input(shape=input_shape)
    
    # first compression
    x = tf.keras.layers.Conv2D(filters, (7,1), strides=1, padding='same', activation='relu')(input_tensor)
    
    # main corpus
    for i in range(n_modules):
        x = resa_red_module(x, filters=filters, kernel_size=kernel_size)
        
    
    # prediction
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    output_tensor = tf.keras.layers.Dense(1, activation='linear')(x)
    
    
    return tf.keras.Model(input_tensor, output_tensor)

def REMNet_mitigation_model(input_length, N, F, batch_size):
    # ResNet classification modification based on "Robust Ultra-wideband Range Error Mitigation with Deep Learning at the Edge"
    r = int(F / 2)
            
    input_shape = ((input_length, 1, 1))
    inputs = layers.Input(shape=input_shape, batch_size=None)
    identity = layers.Conv2D(filters=F, kernel_size=(7, 1), activation='relu', padding="same")(inputs)
    
    for n in range(N):
        conv_2 = layers.Conv2D(filters=F, kernel_size=(3, 1), activation='relu', padding="same")(identity)
        ## SE-block (squeeze-excitation)
        gap_se = layers.GlobalAveragePooling2D()(conv_2)
        dense_1_se = layers.Dense(units=int(F/r), activation="relu")(gap_se)
        dense_2_se = layers.Dense(units=F, activation="sigmoid")(dense_1_se)

        dense_2_se = dense_2_se[:, None, None,:]
        multiply_se = layers.multiply([conv_2, dense_2_se])
        addition_se = layers.add([multiply_se, identity])
        ## reduction block
        reduction_1 = layers.Conv2D(filters=F, kernel_size=(3, 1), strides=(2, 1), padding="same", activation="relu")(addition_se)
        reduction_2 = layers.Conv2D(filters=F, kernel_size=(1, 1), strides=(2, 1), padding="same",activation="relu")(addition_se)
        identity = layers.add([reduction_1, reduction_2])
    
    flatten_final = layers.Flatten()(identity)
    droupout_final = layers.Dropout(0.5)(flatten_final)
    out = layers.Dense(units=1, activation="linear")(droupout_final)
    
    mitigation_model = models.Model(inputs, out)
    
    return mitigation_model


def REMNet_classification_model(input_length, N, F, batch_size):
    # ResNet classification modification based on "Robust Ultra-wideband Range Error Mitigation with Deep Learning at the Edge"
    r = int(F / 2)
            
    input_shape = ((input_length, 1, 1))
    inputs = layers.Input(shape=input_shape, batch_size=None)
    identity = layers.Conv2D(filters=F, kernel_size=(7, 1), activation='relu', padding="same")(inputs)
    
    for n in range(N):
        conv_2 = layers.Conv2D(filters=F, kernel_size=(3, 1), activation='relu', padding="same")(identity)
        ## SE-block (squeeze-excitation)
        gap_se = layers.GlobalAveragePooling2D()(conv_2)
        dense_1_se = layers.Dense(units=int(F/r), activation="relu")(gap_se)
        dense_2_se = layers.Dense(units=F, activation="sigmoid")(dense_1_se)

        dense_2_se = dense_2_se[:, None, None,:]
        multiply_se = layers.multiply([conv_2, dense_2_se])
        addition_se = layers.add([multiply_se, identity])
        ## reduction block
        reduction_1 = layers.Conv2D(filters=F, kernel_size=(3, 1), strides=(2, 1), padding="same", activation="relu")(addition_se)
        reduction_2 = layers.Conv2D(filters=F, kernel_size=(1, 1), strides=(2, 1), padding="same",activation="relu")(addition_se)
        identity = layers.add([reduction_1, reduction_2])
    
    flatten_final = layers.Flatten()(identity)
    droupout_final = layers.Dropout(0.5)(flatten_final)
    out = layers.Dense(units=1, activation="sigmoid")(droupout_final)
    
    classification_model = models.Model(inputs, out)
    return classification_model

def REMNet_multi_output_model(input_length, N, F, batch_size):
    # # ResNet based on "Robust Ultra-wideband Range Error Mitigation with Deep Learning at the Edge"
    r = int(F / 2)
            
    input_shape = ((input_length, 1, 1))
    inputs = layers.Input(shape=input_shape)
    identity = layers.Conv2D(filters=F, kernel_size=(7, 1), activation='relu', padding="same")(inputs)
    
    for n in range(N):
        conv_2 = layers.Conv2D(filters=F, kernel_size=(3, 1), activation='relu', padding="same")(identity)
        ## SE-block (squeeze-excitation)
        gap_se = layers.GlobalAveragePooling2D()(conv_2)
        dense_1_se = layers.Dense(units=int(F/r), activation="relu")(gap_se)
        dense_2_se = layers.Dense(units=F, activation="sigmoid")(dense_1_se)

        dense_2_se = dense_2_se[:, None, None,:]
        multiply_se = layers.multiply([conv_2, dense_2_se])
        addition_se = layers.add([multiply_se, identity])
        ## reduction block
        reduction_1 = layers.Conv2D(filters=F, kernel_size=(3, 1), strides=(2, 1), padding="same", activation="relu")(addition_se)
        reduction_2 = layers.Conv2D(filters=F, kernel_size=(1, 1), strides=(2, 1), padding="same",activation="relu")(addition_se)
        identity = layers.add([reduction_1, reduction_2])

    
    flatten_final = layers.Flatten()(identity)
    droupout_final = layers.Dropout(0.5)(flatten_final)
    out_reg = layers.Dense(units=1, activation="linear")(droupout_final)
    out_class = layers.Dense(units=1, activation="sigmoid")(droupout_final)

    class_reg_model = models.Model(inputs, [out_class, out_reg])
    
    return class_reg_model


def create_SVR_model(kernel="rbf", epsilon=0.1, gamma="scale"):
    return svm.SVR(kernel=kernel, epsilon=epsilon, gamma=gamma)


def create_SVC_model(kernel="rbf", gamma="scale"):
    return svm.SVC(kernel=kernel, gamma=gamma)


def init_REMNet(patience=10, batch_size=1, cir_len=152, 
                start_idx=6, every_xth_item=10, N=3, F=16):
    remnet = convolutional_neural_network_mitigation("REMNet", patience, 
                                                     batch_size, cir_len, 
                                                     cases, correction_methods,
                                                     start_idx, every_xth_item)
    remnet.model_reg = REMNet_mitigation_model(cir_len)
    remnet.model_class = REMNet_classification_model(cir_len, N, F, batch_size)
    return remnet

def init_REMNet_multi_output(patience=10, batch_size=1, cir_len=152, 
                             start_idx=6, every_xth_item=10, N=3, F=16):
    remnet_class_reg_correction_methods = ["Uncorrected", "Corrected", "Multi-output"]
    
    remnet_class_and_reg = REMNet_multi_output("REMNet_multi_output", patience, 
                                                batch_size, cir_len, cases, 
                                                remnet_class_reg_correction_methods,
                                                start_idx, every_xth_item)
    remnet_class_and_reg.model = REMNet_multi_output_model(cir_len, N, F, batch_size)
    return remnet_class_and_reg


def init_CNN(patience=10, batch_size=32, cir_len=152):
    cnn_correction_methods = ["Uncorrected", "Corrected", "DW LOS Estimation"]

    cnn = convolutional_neural_network_mitigation("CNN", patience, batch_size, cir_len, cases, cnn_correction_methods)
    cnn.model_reg = cnn_mitigation_model(cir_len)
    return cnn

def init_SVM(epsilon=0.1, cir_len=152):   
    svm = support_vector_machine("SVM", cases, correction_methods, epsilon, cir_len)
    svm.model_reg = create_SVR_model(epsilon=svm.epsilon)
    svm.model_class = create_SVC_model()
    return svm
