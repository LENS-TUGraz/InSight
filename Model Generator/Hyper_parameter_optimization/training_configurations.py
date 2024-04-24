# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 13:44:58 2023

@author: turtle9
"""
from ML_models.XGBoost.xgboost_model import MLMethodXGRegression, MLMethodXGClassifier
from ML_models.SVM.svm_model import MLMethodSVMRegressor, MLMethodSVMClassifier
from ML_models.REMNet.model import MLMethodREMNetRegression, MLMethodREMNetMultiOutput, MLMethodREMNetClassifier

""" Grid Search """
grid_search_metadata = {
    "remnet": {"reg":{"name_of_best_config":"remnet_reg_minmax",
                    "method":MLMethodREMNetRegression,
                    "fixed_config": {"scaler":"minmaxmatrix", "quantization":"_full_integer", "batch_size":100, "epochs":150}
                    },
            "class":{"name_of_best_config":"remnet_class_minmax",
                        "method":MLMethodREMNetClassifier,
                        "fixed_config": {"scaler":"minmaxmatrix", "quantization":"_full_integer", "batch_size":800, "epochs":150}
                        },
            "mo":{"name_of_best_config":"remnet_multioutput_minmax",
                    "method":MLMethodREMNetMultiOutput,
                    "fixed_config": {"scaler":"minmaxmatrix", "quantization":"_full_integer", "epochs":150}
                    },
        },
        "svm": {"reg":{"name_of_best_config":"svm_regressor_minmax",
                   "method":MLMethodSVMRegressor,
                   "fixed_config":{"scaler":"minmax", "svm_type":3, "kernel":2, "features":['f1','f2','f3','f4','f5','f6', 'f8','f11','f7']}
                   },
            "class":{"name_of_best_config":"svm_classifier_minmax",
                     "method":MLMethodSVMClassifier,
                     "fixed_config":{"scaler":"minmax", "svm_type":0, "kernel":2, "features":['f1','f2','f3','f4','f5','f6', 'f8','f11','f7']}
                     },
        },
         "xgboost": {"reg":{"method":MLMethodXGRegression,
                       "fixed_config":{"scaler":"minmax", "regression_objective":"reg:squarederror", "cir_field":"cir", "gamma":0.5, "subsample":0.6, "eta":0.1}
                       },
                "class":{"method":MLMethodXGClassifier,
                         "fixed_config":{"scaler":"minmax", "classification_objective":"binary:logistic", "cir_field":"cir", "gamma":0, "subsample":0.6, "eta":0.1}
                         },
                },
}
