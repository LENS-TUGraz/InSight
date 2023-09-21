# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 13:44:58 2023

@author: turtle9
"""
from ML_models.xgboost_model import MLMethodXGRegression, MLMethodXGClassifier

""" Grid Search """
grid_search_metadata = {
    "xgboost": {"reg":{"method":MLMethodXGRegression,
                       "fixed_config":{"scaler":"minmax", "regression_objective":"reg:squarederror", "cir_field":"cir", "gamma":0.5, "subsample":0.6, "eta":0.1}
                       },
                "class":{"method":MLMethodXGClassifier,
                         "fixed_config":{"scaler":"minmax", "classification_objective":"binary:logistic", "cir_field":"cir", "gamma":0, "subsample":0.6, "eta":0.1}
                         },
                },
}
