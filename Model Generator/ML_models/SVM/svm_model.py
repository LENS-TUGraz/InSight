#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 12:55:29 2022

@author: michi
"""
import pickle as p
from fastcore.transform import Pipeline
from fastcore.transform import *
from ML_models.SVM.features import calc_features
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.svm import LinearSVC, SVC, SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from gplearn.genetic import SymbolicClassifier
from sklearn.decomposition import PCA
from libsvm.svmutil import *
from sklearn.metrics import r2_score
from ML_models.mlmodelbase import MLMethodBase
from dataclasses import dataclass
from abc import abstractmethod
import numpy as np
from ctypes import *
from ML_models.SVM.libsvm_converter import libsvm_converter
import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
#import treelite

def calc_num_parameters_SVM(model):
    return len(model.get_sv_coef())

#@Transform
#def transform_calculate_svc_features(_df): return calc_features(_df)
class Calculate_Features(Transform):
    def setup(self, config, train_setup): 
        self.cir_start = config['cir_start']
        self.cir_end = config['cir_end']
        self.n_measurements = config["n_measurements"]
    def encodes(self, _df): 
        return calc_features(_df, self.cir_start, self.cir_end, self.n_measurements)

# @Transform
# def transform_calculate_svc_features_kurtosis(_df): return _df['f_otherfeature'] = calc_other_feature()
class Transform_SVM_Classifier(Transform):
    def setup(self, items, train_setup): self.features = items['features']
    def encodes(self, _df):
        # features = _df[self.features + ["setup_id"] + ["NLOS"]]
        # features = features.groupby(["setup_id"]).mean()
        # features = features.reset_index(drop=True)
        # x = features[self.features].values
        # y = features["NLOS"].values
        
        x = _df[self.features].values
        y = _df['NLOS'].values
        return x, y
    
class Transform_SVM_Regression(Transform):
    def setup(self, items, train_setup): self.features = items['features']
    def encodes(self, _df):
        # features = _df[self.features + ["setup_id", "NLOS", "rng_e"]]
        # features = features.groupby(["setup_id"]).mean()
        # features = features.reset_index(drop=True)
        # x = features[self.features].values
        # y = features["rng_e"].values
        
        x = _df[self.features].values       
        y = _df['rng_e'].values
        return x, y
    
class Transform_Classifier(Transform):
    def setup(self, items, train_setup):
        if "cir_field" in items:
            self.cir_field = items['cir_field']
        else:
            self.cir_field = "cir"
        self.cir_end = int(items['cir_end'])
        self.cir_start = int(items['cir_start'])
    def encodes(self, _df):
        Xtrain = np.stack(_df[self.cir_field].values)[:,self.cir_start:self.cir_end]
        ytrain = _df['NLOS'].values
        return Xtrain, ytrain

class Transform_Regression(Transform):
    def setup(self, items, train_setup):
        if "cir_field" in items:
            self.cir_field = items['cir_field']
        else:
            self.cir_field = "cir"
        self.cir_end = int(items['cir_end'])
        self.cir_start = int(items['cir_start'])
    def encodes(self, _df):
        Xtrain = np.stack(_df[self.cir_field].values)[:,self.cir_start:self.cir_end]
        ytrain = _df['rng_e'].values
        return Xtrain, ytrain    

class TransFitMinMaxMatrixModelBased(BaseEstimator, TransformerMixin):
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
        
        if y is not None:
            y = (y - self.y_min)/(self.y_max - self.y_min)
            return X, y
        
        return X        
    
    def inverse_transform(self, y=None):
        return y * (self.y_max - self.y_min) + self.y_min
    
    def fit_transform(self, X, y=None):
        self.fit(X,y)

        if y is not None:
            X,y =self.transform(X,y)
            return X,y
        X =self.transform(X)
        return X
    
class TransFitMinMaxModelBased(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.x_max = None
        self.x_min = None
        self.y_max = None
        self.y_min = None
        self.scale_x = None
        self.scale_y = None
        # print("MinMax Tab wide Scaler")
        
    def fit(self, X,y=None):
        self.x_max = np.max(X,axis=0)
        self.x_min = np.min(X,axis=0)
        if y is not None:
            self.y_max = np.max(y,axis=0)
            self.y_min = np.min(y,axis=0)
            
        return self
    
    def transform(self, X, y=None):
        X = (X - self.x_min)/(self.x_max - self.x_min)
        
        if y is not None:
            y = (y - self.y_min)/(self.y_max - self.y_min)
            return X, y
        
        return X        
    
    def inverse_transform(self, y=None):
        return y * (self.y_max - self.y_min) + self.y_min
    
    def fit_transform(self, X, y=None):
        self.fit(X,y)

        if y is not None:
            X,y =self.transform(X,y)
            return X,y
        X =self.transform(X)
        return X

@dataclass
class MLMethodSVMClassifier(MLMethodBase):
    scaler: str
    def __init__(self, **params):  
        self.scaler = params['scaler'] 
        super().__init__(name=f"svm",mltype="classification", params=params)
        self.params = params

        self.svm_types = {"C-SVC":0, "nu-SVC":1, "one-class SVM":2, "epsilon-SVR":3, "nu-SVR":4}
        self.kernel_types = {"linear":0, "polynomial":1, "rbf":2, "sigmoid":3, "precomputed kernel":4} 
        
        
        self.data_preprocess_pipeline = Pipeline([Transform_SVM_Classifier])
        self.data_preprocess_pipeline.setup(params,False)        

        
    def fit(self, train, val):
        xtrain,ytrain = self.data_preprocess_pipeline(train)
        xval,yval = self.data_preprocess_pipeline(val)
        if self.scaler == "minmax":
            self.tra = MinMaxScaler()
        else:
            assert(False)
        xtrain = self.tra.fit_transform(xtrain)
        xval = self.tra.transform(xval)
        # print(xtrain)
        gamma = 1 / (xtrain.shape[1] * xtrain.var())
        self.model_ = svm_train(ytrain, xtrain, f"""-s {self.params["svm_type"]} -t {self.params["kernel"]} -c {self.params["C"]} -g {gamma}""")

    
    def predict(self, test):
        xtest, ytest = self.data_preprocess_pipeline(test)
        X = self.tra.transform(xtest)
        y_test_hat = svm_predict([], X, self.model_, "-q")[0]
        return  y_test_hat, ytest

    def fit_scaler(self, train):
        xtrain,ytrain = self.data_preprocess_pipeline(train)
        if self.scaler == "minmax":
            self.tra = TransFitMinMaxModelBased()
        else:
            assert(False)
        self.tra.fit(xtrain)
        
    def save(self, file_path=""):
        svm_type = list(self.svm_types.keys())[self.params["svm_type"]]
        svm_save_model(f"{file_path}.model", self.model_)
        
    def load_model(self, file_path=""):
        self.model_ = svm_load_model(f"{file_path}.model")
        
    def convert_to_lite(self, file_name, num_features):
        libsvm_converter(file_name, self.model_, "_class", num_features)
        
    def get_num_parameters(self):
        return calc_num_parameters_SVM(self.model_)
    
    def get_svm_flops(self, **params):
        # SVC https://sci2s.ugr.es/keel/pdf/specific/articulo/vs04.pdf
        # SVR https://www.mathworks.com/help/stats/understanding-support-vector-machine-regression.html
        squares = 1 * params["n_features"]
        l2_norm = 1 * params["n_features"] - 1
        l2_norm_squared = squares + l2_norm
        vector_subtraction = 1 * params["n_features"] - 1
        sigma_squared = 1
        division_by_sigma = 1
        rbf_kernel_flops = l2_norm_squared + vector_subtraction + sigma_squared + division_by_sigma
            
        return params["n_support_vectors"] * 2 * rbf_kernel_flops    

@dataclass
class MLMethodSVMRegressor(MLMethodBase):
    scaler: str
    def __init__(self, **params):  
        self.scaler = params['scaler'] 
        super().__init__(name="svm",mltype="regression", params=params)
        self.params = params

        self.svm_types = {"C-SVC":0, "nu-SVC":1, "one-class SVM":2, "epsilon-SVR":3, "nu-SVR":4}
        self.kernel_types = {"linear":0, "polynomial":1, "rbf":2, "sigmoid":3, "precomputed kernel":4} 
        
        
        self.data_preprocess_pipeline = Pipeline([Transform_SVM_Regression])
        self.data_preprocess_pipeline.setup(params,False)
        
    def fit(self, train, val):
        xtrain,ytrain = self.data_preprocess_pipeline(train)
        if self.scaler == "minmax":
            self.tra = TransFitMinMaxModelBased()
        else:
            assert(False)
        xtrain, ytrain = self.tra.fit_transform(xtrain, ytrain)
        gamma = 1 / (xtrain.shape[1] * xtrain.var())
        self.model_ = svm_train(ytrain.flatten(), xtrain, f"""-s {self.params["svm_type"]} -t {self.params["kernel"]} -c {self.params["C"]} -g {gamma} -p {self.params["epsilon"]}""")

    
    def predict(self, test):
        xtest, ytest = self.data_preprocess_pipeline(test)
        xtest, ytest = self.tra.transform(xtest, ytest)
        y_test_hat = np.asarray(svm_predict([], xtest, self.model_, "-q")[0])
        return  y_test_hat, ytest, self.tra.inverse_transform(y_test_hat)

    def fit_scaler(self, train):
        xtrain,ytrain = self.data_preprocess_pipeline(train)
        if self.scaler == "minmax":
            self.tra = TransFitMinMaxModelBased()
        else:
            assert(False)
        self.tra.fit(xtrain,ytrain)
        
    def save(self, file_path=""):
        svm_type = list(self.svm_types.keys())[self.params["svm_type"]]
        svm_save_model(f"{file_path}.model", self.model_)
        
    def load_model(self, file_path=""):
        self.model_ = svm_load_model(f"{file_path}.model")
        
        
    def convert_to_lite(self, file_name, num_features):
        libsvm_converter(file_name, self.model_, "_reg", num_features)

    def get_num_parameters(self):
        return calc_num_parameters_SVM(self.model_)
    
    def get_svm_flops(self, **params):
        # SVC https://sci2s.ugr.es/keel/pdf/specific/articulo/vs04.pdf
        # SVR https://www.mathworks.com/help/stats/understanding-support-vector-machine-regression.html
        squares = 1 * params["n_features"]
        l2_norm = 1 * params["n_features"] - 1
        l2_norm_squared = squares + l2_norm
        vector_subtraction = 1 * params["n_features"] - 1
        sigma_squared = 1
        division_by_sigma = 1
        rbf_kernel_flops = l2_norm_squared + vector_subtraction + sigma_squared + division_by_sigma
            
        return params["n_support_vectors"] * 2 * rbf_kernel_flops    