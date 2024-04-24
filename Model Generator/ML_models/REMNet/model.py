#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 11:27:24 2022

@author: michi
"""
import tensorflow as tf
import numpy as np
import pathlib
from sklearn.pipeline import Pipeline as SkPipeline
from fastcore.transform import Pipeline
from fastcore.transform import *
from keras import callbacks
from scikeras.wrappers import KerasClassifier, KerasRegressor
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import r2_score, f1_score
from .ml_models import REMNet_classification_model, REMNet_mitigation_model, REMNet_mitigation_original_model, REMNet_multi_output_model
representative_data = None
from ML_models.mlmodelbase import MLMethodBase
from dataclasses import dataclass
from abc import abstractmethod
import matplotlib.pyplot as plt
import subprocess
import os

def plotHistory(history):
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.show()
    
def get_tf_flops(model_h5_path, batch_size=None):
    # https://stackoverflow.com/questions/49525776/how-to-calculate-a-mobilenet-flops-in-keras
    graph = tf.compat.v1.get_default_graph()
    with graph.as_default():
        model = tf.keras.models.load_model(model_h5_path)
        
        if batch_size is None:
            batch_size = 1
            
        inputs = [tf.TensorSpec([batch_size] + inp.shape[1:], inp.dtype) for inp in model.inputs]
        real_model = tf.function(model).get_concrete_function(inputs)

        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()

        flops = tf.compat.v1.profiler.profile(graph=real_model.graph,
                                              run_meta=run_meta, cmd='op', options=opts)

        return flops.total_float_ops
        
def calc_num_parameters(model):
    trainableParams = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
    nonTrainableParams = np.sum([np.prod(v.get_shape()) for v in model.non_trainable_weights])
    totalParams = trainableParams + nonTrainableParams
    return totalParams

def get_flat_buffer_size(model):
    fb_model = tf.lite.TFLiteConverter.from_keras_model(model).convert()
    tf.lite.experimental.Analyzer.analyze(model_content=fb_model)
    # TODO parse output

def calc_arena_size(file_path):   
    arena_size = -1
    p = subprocess.Popen(["wsl", f"./Hyper_parameter_optimization/find-arena-size", f"{file_path}"]
                         , stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # p = subprocess.Popen([f"./InSight_model_generator/find-arena-size", f"{file_path}"]
    #                      , stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = p.communicate()
    rc = p.returncode
    if rc == 0:
        arena_size = eval(output.decode("utf-8"))["arena_size"]
    else:
        print("stderr: " + error.decode("utf-8"))
        print(output.decode("utf-8"))
    return arena_size

def load_pb(pb):
    with tf.gfile.GFile(pb, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
    return graph

def add_scale_values(string, scale_values, scale_type):
    string += f"float {scale_type} [{str(len(scale_values))}] = {{"
    for idx, val in enumerate(scale_values):
        string += str(val)
        if idx < len(scale_values) - 1:
            string += ", "
    string += "};\n"
    return string

class Transform_Classifier(Transform):
    def setup(self, items, train_setup): 
        self.cir_start = items['cir_start']
        self.cir_end = items['cir_end']
    def encodes(self, _df):
        Xtrain = np.stack(_df['cir'].values)[:,self.cir_start:self.cir_end]
        ytrain = _df['NLOS'].values
        return Xtrain, ytrain

class Transform_Regression(Transform):
    def setup(self, items, train_setup): 
        self.cir_start = items['cir_start']
        self.cir_end = items['cir_end']
    def encodes(self, _df):
        Xtrain = np.stack(_df['cir'].values)[:,self.cir_start:self.cir_end]

        ytrain = _df['rng_e'].values
        return Xtrain, ytrain    
    
class Transform_Multi_Output(Transform):
    def setup(self, items, train_setup):         
        self.cir_start = items['cir_start']
        self.cir_end = items['cir_end']
    def encodes(self, _df):
        Xtrain = np.stack(_df['cir'].values)[:,self.cir_start:self.cir_end]
        ytrain_class = _df['NLOS'].values
        ytrain_reg = _df['rng_e'].values
        return Xtrain, ytrain_class, ytrain_reg


class TransFitMinMaxMatrix(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.x_max = None
        self.x_min = None
        self.y_max = None
        self.y_min = None
        self.scale_x = None
        self.scale_y = None
        print("MinMax Matrix wide Scaler")
        
    def fit(self, X,y=None):
        self.x_max = np.max(X)
        self.x_min = np.min(X)
        if y is not None:
            self.y_max = np.max(y)
            self.y_min = np.min(y)
            
        return self
    
    def transform(self, X, y=None):
        X = (X - self.x_min)/(self.x_max - self.x_min)
        X = X.reshape(len(X), len(X[0]), 1, 1)
        X = tf.convert_to_tensor(X, dtype=tf.float32)
        if y is not None:
            y = (y - self.y_min)/(self.y_max - self.y_min)
            return X, y
        
        return X        
    
    def inverse_transform(self, y=None):
        return y*(self.y_max - self.y_min) + self.y_min
    
    def fit_transform(self, X, y=None):
        self.fit(X,y)

        if y is not None:
            X,y =self.transform(X,y)
            return X,y
        X =self.transform(X)
        return X
    
class TransFitMinMax(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.x_max = None
        self.x_min = None
        self.y_max = None
        self.y_min = None
        self.scale_x = None
        self.scale_y = None
        print("MinMax Tab wide Scaler")
        
    def fit(self, X,y=None):
        self.x_max = np.max(X,axis=0)
        self.x_min = np.min(X,axis=0)
        if y is not None:
            self.y_max = np.max(y,axis=0)
            self.y_min = np.min(y,axis=0)
            
        return self
    
    def transform(self, X, y=None):
        X = (X - self.x_min)/(self.x_max - self.x_min)
        X = X.reshape(len(X), len(X[0]), 1, 1)
        X = tf.convert_to_tensor(X, dtype=tf.float32)
        if y is not None:
            y = (y - self.y_min)/(self.y_max - self.y_min)
            return X, y
        
        return X        
    
    def inverse_transform(self, y=None):
        return y*(self.y_max - self.y_min) + self.y_min
    
    def fit_transform(self, X, y=None):
        self.fit(X,y)

        if y is not None:
            X,y =self.transform(X,y)
            return X,y
        X =self.transform(X)
        return X

class TransFitSTD(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.xs = StandardScaler()
        self.ys = StandardScaler()
        print("Init Trans")
        
    def fit(self, X,y=None):
        self.xs = self.xs.fit(X)
        if y is not None:
            self.ys = self.ys.fit(y)
        return self
    
    
    def transform(self, X, y=None):
        Xtrain = self.xs.transform(X)
        Xtrain = Xtrain.reshape(len(Xtrain), len(Xtrain[0]), 1, 1)
        Xtrain = tf.convert_to_tensor(Xtrain, dtype=tf.float32)
        
        if y is not None:
            ytrain = self.ys.transform(y)
            return Xtrain, ytrain
        
        return Xtrain
    
    def inverse_transform(self, y=None):
        return self.ys.inverse_transform(y)
    
    def fit_transform(self, X, y=None):
        self.fit(X,y)
        Xtrain = self.xs.transform(X)
        Xtrain = Xtrain.reshape(len(Xtrain), len(Xtrain[0]), 1, 1)
        Xtrain = tf.convert_to_tensor(Xtrain, dtype=tf.float32)
        
        if y is not None:
            ytrain = self.ys.transform(y)
            return Xtrain,ytrain
        
        return Xtrain
    
@dataclass
class MLMethodREMNetMultiOutput(MLMethodBase):
    scaler: str
    def __init__(self, **params):
        self.scaler = params['scaler']       
        super().__init__(name="remnet",mltype="multioutput", params=params)
        self.data_preprocess_pipeline = Pipeline([Transform_Multi_Output])
        self.data_preprocess_pipeline.setup(params,False)
        self.regressor = None
        self.params = params
        self.use_early_stopping = params["early_stopping"]

        def build_remnet_multi_output_model(params):
            model = REMNet_multi_output_model(params['cir_end']-params['cir_start'], params['N'], params['F'], params['batch_size'])
            adam = tf.keras.optimizers.Adam(learning_rate=3e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False, name='Adam')
            model.compile(adam, loss=[tf.keras.losses.BinaryCrossentropy(), "mean_absolute_error"])
            return model
        
        self.build_model = build_remnet_multi_output_model
    
    def fit(self, train, val):
        xtrain,y_train_class, y_train_reg = self.data_preprocess_pipeline(train)
        xval,y_val_class, y_val_reg = self.data_preprocess_pipeline(val)
        if self.scaler == "minmax":
            self.tra = TransFitMinMax()
        elif self.scaler == "minmaxmatrix":
            self.tra = TransFitMinMaxMatrix()
        else:
            assert(False)
        print(f"{len(xtrain)=}")
        
        plt.figure()
        plt.title("CIR before scaling")
        plt.plot(xval[0])
        plt.show()
        
        X, y_reg = self.tra.fit_transform(xtrain, y_train_reg.reshape(-1,1))
        x_val, y_val_reg = self.tra.transform(xval,y_val_reg.reshape(-1,1))
        # self.regressor.fit(xtrain,ytrain,validation_data=(xval,yval)) #

        plt.figure()
        plt.title("CIR after scaling")
        plt.plot(np.asarray(xval[0][:][:][:]).reshape(-1,1))
        plt.show()
        
        
        self.model = self.build_model(self.params)
        self.data = X
        
        cbs = []
        if self.use_early_stopping:
            early_stopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1,mode='auto', baseline=None, restore_best_weights=True)
            cbs=[early_stopping]
        self.history = self.model.fit(X, [y_train_class, y_reg], validation_data=(x_val, [y_val_class, y_val_reg]), callbacks=cbs, batch_size=self.params['batch_size'], epochs=self.params['epochs'], verbose=1)
        plotHistory(self.history)

    def predict(self, test):
        xtest,y_test_class, y_test_reg = self.data_preprocess_pipeline(test)
        xtest, y_test_reg = self.tra.transform(xtest, y_test_reg.reshape(-1,1))

        y_class_hat, y_reg_hat = self.model.predict(xtest)
        y_class_hat = y_class_hat.round()

        return y_reg_hat, y_test_reg, self.tra.inverse_transform(y_reg_hat), y_class_hat.flatten(), y_test_class


    def convert_saved_model_to_tflite(self, file_path, file_name, quantization=None):
        tflite_models_dir = pathlib.Path(f"{file_path}_lite/")
        tflite_models_dir.mkdir(exist_ok=True, parents=True)
        converter = tf.lite.TFLiteConverter.from_saved_model(f"{file_path}/{file_name}")
        if not quantization:
            # Save the unquanitzed model:
            save_path = f"{file_name}.tflite"
            tflite_normal_model = converter.convert()
            tflite_model__normal_file = tflite_models_dir/save_path
            tflite_model__normal_file.write_bytes(tflite_normal_model)
        elif "full_integer" in quantization:
            # Save the full integer quantized model:
            save_path = f"{file_name}{quantization}.tflite"
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = self.representative_dataset
            tflite_quant_model = converter.convert()
            tflite_model_quant_file = tflite_models_dir/save_path
            tflite_model_quant_file.write_bytes(tflite_quant_model)
        return save_path
    
    def representative_dataset(self):
        for data in tf.data.Dataset.from_tensor_slices((self.data)).batch(1).take(100):
            yield [tf.dtypes.cast(data, tf.float32)]   
        
    def plot_learning_curve(self):
        history_dict = self.history.history
        loss_values = history_dict['loss']
        val_loss_values = history_dict['val_loss']
        epochs = range(1, len(loss_values) + 1)

        fig, ax = plt.subplots()

        ax.plot(epochs, loss_values, label='Training loss')
        ax.plot(epochs, val_loss_values, label='Validation loss')
        ax.set_title('MO Training & Validation Loss', fontsize=16)
        ax.set_xlabel('Epochs', fontsize=16)
        ax.set_ylabel('Mean AbsoluteError', fontsize=16)
        ax.legend()
        plt.show()
    
    def fit_scaler(self, train):
        xtrain,y_train_class, y_train_reg = self.data_preprocess_pipeline(train)
        if self.scaler == "minmax":
            self.tra = TransFitMinMax()
        elif self.scaler == "minmaxmatrix":
            self.tra = TransFitMinMaxMatrix()
        else:
            assert(False)
        self.tra.fit(xtrain,y_train_reg)
        
    def save(self, file_path=""):
        self.model.save(f"{file_path}")
        # df_normalization = pd.DataFrame({"x_min": self.tra.x_min, "x_max":self.tra.x_max, "y_min":self.tra.y_min, "y_max":self.tra.y_max})
        # df_normalization.to_pickle(f"{file_name}_normalization_data.pkl")
        
    def load_model(self, file_path=""):
        self.model = tf.keras.models.load_model(f"{file_path}")
        
    def get_num_parameters(self):
        return calc_num_parameters(self.model)
    
    def find_arena_size(self, file_path):   
        return calc_arena_size(file_path)
    
    def get_flash_size(self):
        return get_flat_buffer_size(self.model)
    
    def calc_flops(self, file_path):
        return get_tf_flops(file_path, batch_size=None)


@dataclass
class MLMethodREMNetRegression(MLMethodBase):
    scaler: str
    def __init__(self, **params):
        self.scaler = params['scaler']       
        super().__init__(name="remnet",mltype="regression", params=params)
        self.data_preprocess_pipeline = Pipeline([Transform_Regression])
        self.data_preprocess_pipeline.setup(params,False)
        self.regressor = None
        self.params = params
        self.use_early_stopping = params["early_stopping"]
        
        def build_model(params):
            # params = params['params']
            model = REMNet_mitigation_model(params['cir_end']-params['cir_start'], params['N'], params['F'], params['batch_size'])
            adam = tf.keras.optimizers.Adam(learning_rate=3e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False, name='Adam')
            model.compile(adam, loss="mean_absolute_error") #
            return model
    
        self.build_model = build_model

    
    def fit(self, train, val):
        xtrain,ytrain = self.data_preprocess_pipeline(train)
        xval,yval = self.data_preprocess_pipeline(val)
        if self.scaler == "minmax":
            self.tra = TransFitMinMax()
        elif self.scaler == "minmaxmatrix":
            self.tra = TransFitMinMaxMatrix()
        else:
            assert(False)
        print(f"{len(xtrain)=}")
        
        plt.figure()
        plt.title("CIR before scaling")
        plt.plot(xval[0])
        plt.show()
        
        xtrain,ytrain = self.tra.fit_transform(xtrain, ytrain.reshape(-1,1))
        xval,yval = self.tra.transform(xval, yval.reshape(-1,1))
        
        plt.figure()
        plt.title("CIR after scaling")
        plt.plot(np.asarray(xval[0][:][:][:]).reshape(-1,1))
        plt.show()
        
        self.regressor = self.build_model(self.params)
        self.data = xtrain
        cbs = []
        if self.use_early_stopping:
            early_stopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1,mode='auto', baseline=None, restore_best_weights=True)
            cbs=[early_stopping]
        self.history= self.regressor.fit(xtrain,ytrain,validation_data=(xval,yval), callbacks=cbs, verbose=1, batch_size=self.params['batch_size'], epochs=self.params['epochs'], shuffle=False) 
        # history = self.regressor.model_.history
        plotHistory(self.history)
        
    def predict(self, test):
        xtest,ytest = self.data_preprocess_pipeline(test)
        xtest,ytest = self.tra.transform(xtest,ytest.reshape(-1,1))
        
        y_test_hat = self.regressor.predict(xtest)
        return y_test_hat, ytest, self.tra.inverse_transform(y_test_hat)
    
    def fit_scaler(self, train):
        xtrain,ytrain = self.data_preprocess_pipeline(train)
        if self.scaler == "minmax":
            self.tra = TransFitMinMax()
        elif self.scaler == "minmaxmatrix":
            self.tra = TransFitMinMaxMatrix()
        else:
            assert(False)
        self.tra.fit(xtrain,ytrain)
        
    def score(self, test):
        pass
        # self.predict(test):
        # xtest,ytest = self.data_preprocess_pipeline(test)
        # xtest,ytest = self.tra.transform(xtest,ytest.reshape(-1,1))
        # score = self.regressor.score(xtest,ytest)
        # print(f"{self.name} score: {score}")
        # return score
        
    def save(self, file_path=""):
        self.regressor.save(f"{file_path}")
        
    def load_model(self, file_path=""):
        self.regressor = tf.keras.models.load_model(f"{file_path}")
        
    def convert_saved_model_to_tflite(self, file_path, file_name, quantization=None):
        tflite_models_dir = pathlib.Path(f"{file_path}_lite/")
        tflite_models_dir.mkdir(exist_ok=True, parents=True)
        converter = tf.lite.TFLiteConverter.from_saved_model(f"{file_path}/{file_name}")
        if not quantization:
            # Save the unquanitzed model:
            save_path = f"{file_name}.tflite"
            tflite_normal_model = converter.convert()
            tflite_model__normal_file = tflite_models_dir/save_path
            tflite_model__normal_file.write_bytes(tflite_normal_model)
        elif "full_integer" in quantization:
            # Save the full integer quantized model:
            save_path = f"{file_name}{quantization}.tflite"
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = self.representative_dataset
            tflite_quant_model = converter.convert()
            tflite_model_quant_file = tflite_models_dir/save_path
            tflite_model_quant_file.write_bytes(tflite_quant_model)
        return save_path
    
    def representative_dataset(self):
        for data in tf.data.Dataset.from_tensor_slices((self.data)).batch(1).take(100):
            yield [tf.dtypes.cast(data, tf.float32)]   
            
    def plot_learning_curve(self):
        history_dict = self.regressor.history_
        loss_values = history_dict['loss']
        val_loss_values = history_dict['val_loss']
        epochs = range(1, len(loss_values) + 1)

        fig, ax = plt.subplots()

        ax.plot(epochs, loss_values, label='Training loss')
        ax.plot(epochs, val_loss_values, label='Validation loss')
        ax.set_title('Regression Training & Validation Loss', fontsize=16)
        ax.set_xlabel('Epochs', fontsize=16)
        ax.set_ylabel('Mean AbsoluteError', fontsize=16)
        ax.legend()
        plt.show()
        
    def get_num_parameters(self):
        return calc_num_parameters(self.regressor)
    
    def find_arena_size(self, file_path):   
        return calc_arena_size(file_path)
    
    def get_flash_size(self):
        return get_flat_buffer_size(self.regressor)
    
    def get_tf_flops(self, file_path):
        # https://stackoverflow.com/questions/49525776/how-to-calculate-a-mobilenet-flops-in-keras
        graph = tf.compat.v1.get_default_graph()
        with graph.as_default():
            model = tf.keras.models.load_model(file_path)
            
            if batch_size is None:
                batch_size = 1
                
            inputs = [tf.TensorSpec([batch_size] + inp.shape[1:], inp.dtype) for inp in model.inputs]
            real_model = tf.function(model).get_concrete_function(inputs)

            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()

            flops = tf.compat.v1.profiler.profile(graph=real_model.graph,
                                                run_meta=run_meta, cmd='op', options=opts)

            return flops.total_float_ops
        
@dataclass
class MLMethodREMNetClassifier(MLMethodBase):
    scaler: str
    def __init__(self, **params):
        self.scaler = params['scaler']       
        super().__init__(name="remnet",mltype="classification", params=params)
        self.data_preprocess_pipeline = Pipeline([Transform_Classifier])
        self.data_preprocess_pipeline.setup(params,False)
        self.classifier = None
        self.params = params
        # self.model = None
        self.tra = None
        self.use_early_stopping = params["early_stopping"]

        def build_model(params):
            # params = params['params']
            model = REMNet_classification_model(self.params['cir_end']-self.params['cir_start'], self.params['N'], self.params['F'], self.params['batch_size'])
            adam = tf.keras.optimizers.Adam(learning_rate=3e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False, name='Adam')
            model.compile(adam, loss="binary_crossentropy") #
            return model
        
        self.build_model = build_model
       
    def fit(self, train, val):
        xtrain,ytrain = self.data_preprocess_pipeline(train)
        xval,yval = self.data_preprocess_pipeline(val)
        if self.scaler == "minmax":
            self.tra = TransFitMinMax()
        elif self.scaler == "minmaxmatrix":
            self.tra = TransFitMinMaxMatrix()
        else:
            assert(False)
        xtrain = self.tra.fit_transform(xtrain)
        xval = self.tra.transform(xval)

        self.classifier = self.build_model(self.params)
        self.data = xtrain
        cbs = []
        if self.use_early_stopping:
            early_stopping_remnet_classifier = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto', baseline=None, restore_best_weights=True)
            cbs=[early_stopping_remnet_classifier]
        self.history = self.classifier.fit(xtrain, ytrain, validation_data=(xval,yval), callbacks=cbs, verbose=0, batch_size=self.params['batch_size'], epochs=self.params['epochs'], shuffle=False)
        # history = self.classifier.model_.history
        plotHistory(self.history)

    def predict(self, test):
        xtest,y_test = self.data_preprocess_pipeline(test)
        xtest = self.tra.transform(xtest)
        y_test_hat = self.classifier.predict(xtest)
        y_test_hat = np.round(y_test_hat)
        return y_test_hat.flatten(), y_test
    
    def fit_scaler(self, train):
        xtrain,ytrain = self.data_preprocess_pipeline(train)
        if self.scaler == "minmax":
            self.tra = TransFitMinMax()
        elif self.scaler == "minmaxmatrix":
            self.tra = TransFitMinMaxMatrix()
        else:
            assert(False)
        self.tra.fit(xtrain)
        
    def score(self, test):
        pass
        # xtest,ytest = self.data_preprocess_pipeline(test)
        # xtest = self.tra.transform(xtest)
        # score = self.classifier.score(xtest,ytest)
        # print(f"{self.name} f1 score: {score}")
        # return score
        
    def save(self, file_path=""):
        self.classifier.save(f"{file_path}")
        
    def load_model(self, file_path=""):
        self.classifier = tf.keras.models.load_model(f"{file_path}")
            
    def convert_saved_model_to_tflite(self, file_path, file_name, quantization=None):
        tflite_models_dir = pathlib.Path(f"{file_path}_lite/")
        tflite_models_dir.mkdir(exist_ok=True, parents=True)
        converter = tf.lite.TFLiteConverter.from_saved_model(f"{file_path}/{file_name}")
        if not quantization:
            # Save the unquanitzed model:
            save_path = f"{file_name}.tflite"
            tflite_normal_model = converter.convert()
            tflite_model__normal_file = tflite_models_dir/save_path
            tflite_model__normal_file.write_bytes(tflite_normal_model)
        elif "full_integer" in quantization:
            # Save the full integer quantized model:
            save_path = f"{file_name}{quantization}.tflite"
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = self.representative_dataset
            tflite_quant_model = converter.convert()
            tflite_model_quant_file = tflite_models_dir/save_path
            tflite_model_quant_file.write_bytes(tflite_quant_model)
            tflite_model_quant_file
        return save_path
    
    def representative_dataset(self):
        for data in tf.data.Dataset.from_tensor_slices((self.data)).batch(1).take(100):
            yield [tf.dtypes.cast(data, tf.float32)]   
            
    def plot_learning_curve(self):
        history_dict = self.classifier.history_
        loss_values = history_dict['loss']
        val_loss_values = history_dict['val_loss']
        epochs = range(1, len(loss_values) + 1)

        fig, ax = plt.subplots()

        ax.plot(epochs, loss_values, label='Training loss')
        ax.plot(epochs, val_loss_values, label='Validation loss')
        ax.set_title('Classification Training & Validation Loss', fontsize=16)
        ax.set_xlabel('Epochs', fontsize=16)
        ax.set_ylabel('Binary Crossentropy', fontsize=16)
        ax.legend()
        plt.show()
        
    def get_num_parameters(self):
        return calc_num_parameters(self.classifier)
    
    def find_arena_size(self, file_path):   
        return calc_arena_size(file_path)
    
    def get_flash_size(self):
        return get_flat_buffer_size(self.classifier)
    
    def get_tf_flops(self, file_path):
        # https://stackoverflow.com/questions/49525776/how-to-calculate-a-mobilenet-flops-in-keras
        graph = tf.compat.v1.get_default_graph()
        with graph.as_default():
            model = tf.keras.models.load_model(file_path)
            
            if batch_size is None:
                batch_size = 1
                
            inputs = [tf.TensorSpec([batch_size] + inp.shape[1:], inp.dtype) for inp in model.inputs]
            real_model = tf.function(model).get_concrete_function(inputs)

            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()

            flops = tf.compat.v1.profiler.profile(graph=real_model.graph,
                                                run_meta=run_meta, cmd='op', options=opts)

            return flops.total_float_ops
    
    