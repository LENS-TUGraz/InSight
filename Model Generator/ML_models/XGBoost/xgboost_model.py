# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 12:55:29 2022

@author: michi
"""
import pickle as p
from fastcore.transform import Pipeline
from fastcore.transform import *
from Dataset_preprocessing.preprocessing_pipeline.features import calc_features
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
from ML_models.XGBoost.xgboost_converter import xgboost_converter
import pandas as pd


def calc_num_parameters_XGBoost(model):
    tree = model.get_booster().trees_to_dataframe()
    return tree.groupby("Tree").size().sum()
    
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


from xgboost import XGBClassifier, XGBRegressor
from xgboost import plot_tree
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin

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
class MLMethodXGRegression(MLMethodBase):
    scaler: str
    def create_model(self):
        self.model = XGBRegressor(max_depth=int(self.params["max_depth"]), 
                                  eta=float(self.params["eta"]), 
                                  subsample=float(self.params["subsample"]), 
                                  gamma=float(self.params["gamma"]), 
                                  objective=self.params["regression_objective"],
                                  n_estimators=int(self.params["n_estimators"]),
                                   early_stopping_rounds=10)
        
    def __init__(self, **params):
        self.scaler = params['scaler'] 
        super().__init__(name="xgboost",mltype="regression", params=params)
        # self.data_preprocess_pipeline = Pipeline([Transform_SVM_Regression])
        self.data_preprocess_pipeline = Pipeline([Transform_Regression()])
        self.data_preprocess_pipeline.setup(params,False)
        self.create_model()
        
    def fit(self, train, val):
        self.create_model()
        xtrain,ytrain = self.data_preprocess_pipeline(train)
        xval,yval = self.data_preprocess_pipeline(val)
        if self.scaler == "minmax":
            self.tra = TransFitMinMaxModelBased()
        elif self.scaler == "minmaxmatrix":
            self.tra = TransFitMinMaxMatrixModelBased()
        else:
            assert(False)
            
        plt.figure()
        plt.title("CIR before scaling")
        plt.plot(xval[0])
        plt.show()
        
        xtrain,ytrain = self.tra.fit_transform(xtrain,ytrain)
        xval,yval = self.tra.transform(xval,yval)
        
        plt.figure()
        plt.title("CIR after scaling")
        plt.plot(np.asarray(xval[0][:][:][:]).reshape(-1,1))
        plt.show()    
    
        self.model.fit(xtrain,ytrain, eval_set=[(xtrain,ytrain), (xval, yval)], verbose=False)
        self.params["best_n_estimators"] = self.model.best_ntree_limit
    
    def predict(self, test):
        xtest, ytest = self.data_preprocess_pipeline(test)
        # xtest = self.tra_x.transform(xtest)
        xtest,ytest = self.tra.transform(xtest,ytest)
        # xtest = self.tra_x.transform(xtest)
        # ytest = self.tra_y.transform(ytest.reshape(-1, 1))
        y_test_hat = self.model.predict(xtest)
        return  y_test_hat, ytest, self.tra.inverse_transform(y_test_hat.reshape(-1, 1))
            
    def fit_scaler(self, train):
        xtrain,ytrain = self.data_preprocess_pipeline(train)
        if self.scaler == "minmax":
            self.tra = TransFitMinMaxModelBased()
        elif self.scaler == "minmaxmatrix":
            self.tra = TransFitMinMaxMatrixModelBased()
        else:
            assert(False)
        self.tra.fit(xtrain,ytrain)
        
    def set_scaler(self, file_path):
        if self.scaler == "minmax":
            self.tra = TransFitMinMaxModelBased()
        elif self.scaler == "minmaxmatrix":
            self.tra = TransFitMinMaxMatrixModelBased()
        else:
            assert(False)
        val_dict = pd.read_pickle(f"{file_path}.pkl")
        self.tra.x_min = val_dict.x_min.values
        self.tra.x_max = val_dict.x_max.values
        self.tra.y_min = val_dict.y_min.unique()[0]
        self.tra.y_max = val_dict.y_max.unique()[0]
        
    def plot_tree(self):
        plot_tree(self.model, rankdir="LR")
        
    def plot_importance(self, plottype="ordered", ax=None):
        if plottype == "ordered":
            plot_importance(self.model,ax)
        else:
            ax.bar(range(len(self.model.feature_importances_)), self.model.feature_importances_)
        plt.show()
        
    def save(self, file_name):
        self.model.save_model(f"{file_name}.model")
        df_normalization = pd.DataFrame({"x_min": self.tra.x_min, "x_max":self.tra.x_max, "y_min":self.tra.y_min, "y_max":self.tra.y_max})
        df_normalization.to_pickle(f"{file_name}_normalization_data.pkl")
        
    def load_model(self, file_name):
        self.model.load_model(f"{file_name}.model")
        
    def convert_to_lite(self, file_name, n_classes=0, add_scaling_values=False):
        if add_scaling_values and self.scaler == "minmax":
            scale_min = self.tra.x_min
            scale_max = self.tra.x_max
        else:
            scale_min = []
            scale_max = []
        xgboost_converter(file_name, self.model, self.mltype, n_classes, scale_min, scale_max)

    # def create_config_file(file_name, self.mltype, scale_min, scale_max):
    #     if len(scale_min) > 0:
    #         string = add_scale_values(string, scale_min, "min")
    #     if len(scale_max) > 0:
    #         string = add_scale_values(string, scale_max, "max")
            
    #     f = open(f"{file_name}", "w")
    #     f.write(string)
    #     f.close()
  
    def plot_learning_curve(self):
        results = self.model.evals_result()
        epochs = len(results['validation_0']['rmse'])
        x_axis = range(0, epochs)

        fig, ax = plt.subplots()
        ax.plot(x_axis, results['validation_0']['rmse'], label='Train')
        ax.plot(x_axis, results['validation_1']['rmse'], label='Val')
        ax.legend()
        plt.ylabel('RMSE')
        plt.title('XGBoost RMSE')
        plt.show()
        
    def get_num_parameters(self):
        return calc_num_parameters_XGBoost(self.model)
    
    def get_xgboost_flops(self, **params):
        leaf_addition = 1
        tree_flops = (params["max_depth"] + leaf_addition) * params["n_estimators"]
        output_flops = 1
            
        total_flops = tree_flops + output_flops
        return total_flops


from xgboost import plot_importance
@dataclass
class MLMethodXGClassifier(MLMethodBase):
    scaler: str
    def create_model(self):
        self.model = XGBClassifier(max_depth=int(self.params["max_depth"]), 
                                   eta=float(self.params["eta"]), 
                                   subsample=float(self.params["subsample"]), 
                                   gamma=float(self.params["gamma"]), 
                                   objective=self.params["classification_objective"],
                                   n_estimators=int(self.params["n_estimators"]),
                                    early_stopping_rounds=10)
    
    def __init__(self, **params):
        self.scaler = params['scaler'] 
        super().__init__(name="xgboost",mltype="classification", params=params)
        print(f"Init XGBoostClassifier {params=}")
        # self.data_preprocess_pipeline = Pipeline([Transform_SVM_Classifier])
        self.data_preprocess_pipeline = Pipeline([Transform_Classifier()])
        self.data_preprocess_pipeline.setup(params,False)
        self.create_model()

    def fit(self, train, val):
        self.create_model()
        xtrain,ytrain = self.data_preprocess_pipeline(train)
        xval,yval = self.data_preprocess_pipeline(val)
        if self.scaler == "minmax":
            self.tra = TransFitMinMaxModelBased()
        elif self.scaler == "minmaxmatrix":
            self.tra = TransFitMinMaxMatrixModelBased()
        else:
            assert(False)
        xtrain = self.tra.fit_transform(xtrain)
        xval = self.tra.transform(xval)
        
        # xtrain[-1] = np.zeros(len(xtrain[-1]))
        # ytrain[-1] = ytrain.max() + 1
        self.model.fit(xtrain, ytrain, eval_set=[(xtrain,ytrain), (xval, yval)], verbose=False)
        self.params["best_n_estimators"] = self.model.best_ntree_limit

    def predict(self, test):
        xtest,ytest = self.data_preprocess_pipeline(test)
        # if self.scaler == "minmax":
        #     self.tra = MinMaxScaler()
        # else:
        #     assert(False)
        # xtest = self.tra.fit_transform(xtest)
        xtest = self.tra.transform(xtest)
        y_test_hat = self.model.predict(xtest)
        
        return y_test_hat,ytest
    
    def fit_scaler(self, train):
        xtrain,ytrain = self.data_preprocess_pipeline(train)
        if self.scaler == "minmax":
            self.tra = TransFitMinMaxModelBased()
        elif self.scaler == "minmaxmatrix":
            self.tra = TransFitMinMaxMatrixModelBased()
        else:
            assert(False)
        self.tra.fit(xtrain)
        
    def set_scaler(self, file_path):
        if self.scaler == "minmax":
            self.tra = TransFitMinMaxModelBased()
        elif self.scaler == "minmaxmatrix":
            self.tra = TransFitMinMaxMatrixModelBased()
        else:
            assert(False)
        val_dict = pd.read_pickle(f"{file_path}.pkl")
        self.tra.x_min = val_dict.x_min.values
        self.tra.x_max = val_dict.x_max.values
        self.tra.y_min = val_dict.y_min.unique()[0]
        self.tra.y_max = val_dict.y_max.unique()[0]
            
    def plot_tree(self):
        plot_tree(self.model, rankdir="LR")
        
    def plot_importance(self, plottype="ordered", ax=None):
        if plottype == "ordered":
            plot_importance(self.model,ax)
        else:
            ax.bar(range(len(self.model.feature_importances_)), self.model.feature_importances_)
        plt.show()
    
    def plot_tree(self):
        plot_tree(self.model, rankdir="LR")
        
    def save(self, file_name):
        self.model.save_model(f"{file_name}.model")
        df_normalization = pd.DataFrame({"x_min": self.tra.x_min, "x_max":self.tra.x_max, "y_min":self.tra.y_min, "y_max":self.tra.y_max})
        df_normalization.to_pickle(f"{file_name}_normalization_data.pkl")
        
    def load_model(self, file_name):
        self.model.load_model(f"{file_name}.model")
        
    def convert_to_lite(self, file_name, n_classes, add_scaling_values=False):
        if add_scaling_values and self.scaler == "minmax":
            scale_min = self.tra.x_min
            scale_max = self.tra.x_max
        else:
            scale_min = []
            scale_max = []
        xgboost_converter(file_name, self.model, "classification", n_classes, scale_min, scale_max)
        
    def plot_learning_curve(self):
        results = self.model.evals_result()
        epochs = len(results['validation_0']['rmse'])
        x_axis = range(0, epochs)

        fig, ax = plt.subplots()
        ax.plot(x_axis, results['validation_0']['rmse'], label='Train')
        ax.plot(x_axis, results['validation_1']['rmse'], label='Val')
        ax.legend()
        plt.ylabel('Log Loss')
        plt.title('XGBoost Log Loss')
        plt.show()
        
    def get_num_parameters(self):
        return calc_num_parameters_XGBoost(self.model)
    
    def get_xgboost_flops(self, **params):
        leaf_addition = 1
        tree_flops = (params["max_depth"] + leaf_addition) * params["n_estimators"]
        output_flops = 1
            
        total_flops = tree_flops + output_flops
        return total_flops

