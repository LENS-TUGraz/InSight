# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 09:26:33 2022

@author: turtle9
"""

""" LOAD PRE-PROCESSED DATA """
import pickle as p
import os

datasets_inital_processed = []
file_name = "train_data"
with open(f"Datasets/preprocessed_datasets/{file_name}.pkl", 'rb') as f:
   datasets_inital_processed.append(p.load(f))
   
file_name = "test_data"
if os.path.isfile(f"Datasets/preprocessed_datasets/{file_name}.pkl"):
    with open(f"Datasets/preprocessed_datasets/{file_name}.pkl", 'rb') as f:
        datasets_inital_processed.append(p.load(f))
   
#%%
""" HYPER-PARAMETER OPTIMIZATION """
import logging

log = logging.getLogger('MLHPE')
if not log.hasHandlers():
    log.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(name)s:%(levelname)s:%(message)s')
    ch.setFormatter(formatter)
    log.addHandler(ch)
    
import matplotlib.pyplot as plt
import itertools
import pandas as pd
import numpy as np
import tensorflow as tf
from fastcore.transform import Pipeline
from sklearn.metrics import balanced_accuracy_score, accuracy_score, precision_score, recall_score, f1_score,confusion_matrix,ConfusionMatrixDisplay,r2_score,mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from Hyper_parameter_optimization.data_analysis import plot_data_set_distribution
from Hyper_parameter_optimization.data_split import split_dataframe,split_test_val
from Hyper_parameter_optimization.training_configurations import grid_search_metadata
from sklearn.utils.class_weight import compute_sample_weight
import yaml

seed = 100
np.random.seed(seed)
tf.keras.utils.set_random_seed(seed)

def load_yaml(file_name):
    file = {}
    with open(f"{file_name}.yaml") as f:
        file = yaml.load(f, Loader=yaml.SafeLoader)
    return file

config = load_yaml('./Hyper_parameter_optimization/hyper_parameter_optimization_config')
models_to_run = config["models_to_run"]
grid_search_config = config["grid_search_config"]
training_configuration = config["training_configuration"]

df_grid_search = pd.DataFrame()
for model, mltype in models_to_run:
    config_combinations = []
    for config_combination in itertools.product(*grid_search_config[model].values()):
        config = grid_search_metadata[model][mltype]["fixed_config"]
        df_kfold_mean = pd.DataFrame()
        config_combinations.append(config_combination)
        print(f"{config_combination=}")
        for idx, hyperparameter in enumerate(grid_search_config[model].keys()):
            config[hyperparameter] = config_combination[idx] 
        config["cir_len"] = config["cir_end"] - config["cir_start"]
        print(f"{config}")
        
        if config["cir_len"] <= 0:
            log.warning("CIR length is negative. Skipping configuration step.")
            continue
        
        # Instantiate model
        classifier_approaches = [
            grid_search_metadata[model][mltype]["method"](**config)
            ]

            
        from Dataset_preprocessing.preprocessing_pipeline.preprocessing import Filter_Conditions,  KF_Grouping, Filter_Resampling, ScaleRNGEMinMax, Filter_Environments
        pipeline_filter = Pipeline([
            Filter_Conditions,
            Filter_Resampling,
            ScaleRNGEMinMax,
            KF_Grouping
            ])
        pipeline_filter.setup(training_configuration, train_setup=False)
        
        
        
        ''' Initiate the result dict '''
        if len(datasets_inital_processed) == 1:
            ds_name_1, data_set_1 = datasets_inital_processed[0]
        elif len(datasets_inital_processed) == 2:
            ds_name_1, data_set_1 = datasets_inital_processed[0]
            ds_name_2, data_set_2 = datasets_inital_processed[1]
        else:
            assert False, "Provide one or two datasets!"
        if len(datasets_inital_processed) == 2:
            dff1 = pipeline_filter(data_set_1)
            dff2 = pipeline_filter(data_set_2)
            split_iterator = split_test_val(dff1,dff2,train_size=0.8, seed=seed)
        else:
            dff1 = pipeline_filter(data_set_1)
            dff2 = dff1
            split_iterator = split_dataframe(dff1, training_configuration['kfolds'], setup_name="setup_id", train_size=0.8, validation=True, seed=seed)
        print("---------------------------------------")
        print(f"Training on {ds_name_1} testing on {ds_name_2}")
        
        results = {}
        for it,(train, test, val) in enumerate(split_iterator):            
            dff_train = dff1.loc[train]
            dff_val = dff1.loc[val] 
            dff_test = dff2.loc[test]
            
    
            for mlmod  in classifier_approaches:
                print(f"Evaluate type: {mlmod.name} name:  KF {it}")
                
                ''' Prepare data set for model via the data preprocess pipeline '''
                mlmod.fit(dff_train, dff_val)
     
    
                ''' Fit model via the ml pipeline '''
                result_entry =  {'ds_name_1':ds_name_1,
                                 'ds_name_2':ds_name_2,
                                'params':mlmod.params,
                                 'mltype':mlmod.mltype,
                                 'kf_it':it,
                                 'idx_train':train,
                                 'idx_test':test,
                                 'idx_val':val 
                                 }
                
                if mlmod.mltype == "regression":
                    y_reg_hat_scaled, y_reg_scaled, y_reg_hat = mlmod.predict(dff_test)
                    data = {"y_reg_scaled":y_reg_scaled.flatten(),
                            "y_reg_hat_scaled":y_reg_hat_scaled.flatten(), 
                            "y_reg_hat":y_reg_hat.flatten(),
                            "original_index": dff_test.original_index,
                            "dataset_id": dff_test.dataset_id,
                            "rng_e":dff_test.rng_e}

                    
                    
                    sel = ~np.isnan(y_reg_hat_scaled)
                    result_entry['score_r2'] = r2_score(y_reg_scaled[sel], y_reg_hat_scaled[sel])
                    print(f"R2 Score {result_entry['score_r2'] }")
                    
                    result_entry['score_ns_r2'] = r2_score(dff_test.rng_e.values.reshape(-1,1)[sel], y_reg_hat[sel])
                    print(f"R2 Score {result_entry['score_ns_r2'] }")
                    
                    
                elif mlmod.mltype == "classification":
                    # score_f1 = mlmod.score(dfc.loc[test])
                    y_class_hat, y_class = mlmod.predict(dff_test)
                    
                    data = {"y_class_hat":y_class_hat,
                            "original_index": dff_test.original_index,
                            "dataset_id": dff_test.dataset_id,
                            "y_class":y_class,
                            
                            }
                    if "rng_e" in dff_test:
                        data["rng_e_used"] = dff_test.rng_e
                    

                    if "plot_importance" in dir(mlmod):
                         # mlmod.plot_tree()
                         fig = plt.figure()
                         # fig.set_size_inches(10,40)
                         ax = plt.gca()
                         mlmod.plot_importance(plottype="linear", ax=ax)
                    
                    weights = compute_sample_weight(class_weight="balanced",y=y_class)
                    
                    result_entry['score_f1'] = f1_score(y_class,y_class_hat,sample_weight=weights)
                    result_entry['score_accuracy'] = accuracy_score(y_class,y_class_hat,sample_weight=weights)
                    result_entry['score_precision'] = precision_score(y_class,y_class_hat,sample_weight=weights)
                    result_entry['score_recall'] = recall_score(y_class,y_class_hat,sample_weight=weights)
                    
                    scaler = MinMaxScaler()
                    # weights = scaler.fit_transform(dfc.loc[test].rng_e.abs().values.reshape(-1,1)).flatten()
                    if dff_test.rng_e_scaled.sum() > 0:
                        weights = dff_test.rng_e_scaled
                        
                        wscore = f1_score(y_class,y_class_hat,sample_weight=weights)
                        result_entry['score_wf1'] = wscore
                        
                        wscore = accuracy_score(y_class,y_class_hat,sample_weight=weights)
                        result_entry['score_waccuracy'] = wscore

                        wscore = precision_score(y_class,y_class_hat,sample_weight=weights)
                        result_entry['score_wprecision'] = wscore
                        
                        wscore = recall_score(y_class,y_class_hat,sample_weight=weights)
                        result_entry['score_wrecall'] = wscore
                    else:
                        result_entry['score_wf1'] = 0
                        result_entry['score_waccuracy'] = 0
                        result_entry['score_wprecision'] = 0
                        result_entry['score_wrecall'] = 0
                    # 
                    print(f"F1 Score {result_entry['score_f1'] }")
                    print(f"Accuracy Score {result_entry['score_accuracy'] }")
                    print(f"Precision Score {result_entry['score_precision'] }")
                    print(f"Recall Score {result_entry['score_recall'] }")
                    
                    
                ''' Estimate the memory size'''
                folder = ""
                if "xgboost" in mlmod.name.lower():
                    folder = "xgboost"
                    file_name = f"""{mlmod.name}_{mlmod.mltype}"""
                    if not os.path.exists(f"./Hyper_parameter_optimization/intermediate_results/{folder}"):
                        os.makedirs(f"./Hyper_parameter_optimization/intermediate_results/{folder}")
                    
                mlmod.save(f"""./Hyper_parameter_optimization/intermediate_results/{folder}/{mlmod.name}_{mlmod.mltype}""")
                
                if "xgboost" in mlmod.name.lower():
                    if not os.path.exists(f"./Hyper_parameter_optimization/intermediate_results/{folder}_lite"):
                        os.makedirs(f"./Hyper_parameter_optimization/intermediate_results/{folder}_lite")
                    mlmod.convert_to_lite(f"""./Hyper_parameter_optimization/intermediate_results/{folder}_lite/{mlmod.name}_{mlmod.mltype}.h""", len(dff_test.NLOS.unique()))

                if "xgboost" in mlmod.name.lower():
                    model_size = os.path.getsize(f"""./Hyper_parameter_optimization/intermediate_results/{folder}_lite/{file_name}.h""")
                else:
                    model_size = 0
                        
                result_entry["model_size"] = model_size
                print(f"{model_size = }")
                
                """ Estimate number of parameters """
                ### total params in NN, number of support vectors in SVM, and number of nodes in XGBoost Tree
                n_parameters = mlmod.get_num_parameters()
                result_entry["n_parameters"] = n_parameters
                print(f"{n_parameters = }")

                
                ''' append to results '''
                if model == "svm" and mlmod.mltype == "regression":
                    if len(dff_test) == len(y_reg_hat):
                        dfres = pd.DataFrame(index=dff_test.index, data=data)                           
                    else:
                        dfres = pd.DataFrame(index=dff_test.setup_id.unique(),data=data)
                else:
                    dfres = pd.DataFrame(index=dff_test.index, data=data)
                result_entry['pred'] = dfres
                
                if mlmod.name in results.keys():
                    results[mlmod.name].append(result_entry)
                else:
                    results[mlmod.name] = []
                    results[mlmod.name].append(result_entry)

                        
        output_file_name = f"./Hyper_parameter_optimization/hyper_parameter_output/{model}_{mltype}.pkl"
        if os.path.isfile(output_file_name):
            df_res = pd.read_pickle(output_file_name)
        else:
            df_res = pd.DataFrame()
        for model_name, model_results in results.items():
            # res.append()
            for iteration in model_results:
                iteration['model_name'] = model_name
                df_res = df_res.append(iteration, ignore_index=True)
        df_res.to_pickle(output_file_name)

                        
