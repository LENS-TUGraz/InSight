# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 13:59:46 2023

@author: turtle9
"""

""" MODEL SELECTION """
import logging

log = logging.getLogger('MLSEL')
if not log.hasHandlers():
    log.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(name)s:%(levelname)s:%(message)s')
    ch.setFormatter(formatter)
    log.addHandler(ch)
    
import pandas as pd
import numpy as np
import tensorflow as tf
import json
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
from matplotlib.ticker import LogLocator
from Model_selection.model_selection_helper import *
import yaml
from yaml.loader import SafeLoader
from pymoo.decomposition.asf import ASF

hpo_config = {}
with open('./Hyper_parameter_optimization/hyper_parameter_optimization_config.yaml') as f:
    hpo_config = yaml.load(f, Loader=SafeLoader)
    
config = {}
with open('./Model_selection/model_selection_config.yaml') as f:
    config = yaml.load(f, Loader=SafeLoader)
    
requirements = {}
with open('./user_specific_requirements.yaml') as f:
    requirements = yaml.load(f, Loader=SafeLoader)

models = list(set(np.asarray(hpo_config["models_to_run"])[:, 0]))
grid_search_config = hpo_config["grid_search_config"]
objective_class = config["objective_class"]
objective_reg = config["objective_reg"]

file_path = "Hyper_parameter_optimization/hyper_parameter_output"
output_path = "Model_selection/model_selection_output"


def prepare_data(dfa,x_axis, interesting_params):
    # add params dict values as columns
    dfa[list(dfa.params.iloc[0].keys())] = dfa.apply(lambda a: [a.params[e] for e in dfa.params.iloc[0].keys()], result_type="expand", axis=1)
    # hacky, filter out reg:squarederror for classification objective
    if "xgboost" in dfa.model_name:
        dfa = dfa[dfa.classification_objective == "binary:logistic"]
    # take mean of model params in case of kfold
    dfa = dfa.groupby(list(interesting_params.keys())).mean().reset_index()
    # sort values based on model size    
    dfs = dfa.sort_values(x_axis, ascending=False)
    
    dfs["model_size"] = dfs.model_size / 1000
    return dfs
#%% Calculate memory and estimated runtime, create Pareto Front and filter out unsuitable candidates
df_res = []
for model in models:
    if model == "xgboost":
        max_footprint = requirements["FLASH"]
    else:
        max_footprint = requirements["RAM"]
    max_runtime = requirements["Runtime"]
    log.info(f"Loading user-specific requirements: {requirements}")
    interesting_params = grid_search_config[model]

    dfclass = pd.read_pickle(f"{file_path}/{model}_class.pkl")
    dfreg = pd.read_pickle(f"{file_path}/{model}_reg.pkl")
    
    dfcs = prepare_data(dfclass, "model_size", interesting_params)
    dfrs = prepare_data(dfreg, "model_size", interesting_params)
    
    if model == "xgboost":
        max_footprint = requirements["FLASH"]
    else:
        max_footprint = requirements["RAM"]
    class_df_od = dfcs
    reg_df_od = dfrs
    
    params = list(interesting_params.keys()) #+ ["flops"]
    class_df_od_f = class_df_od
    class_df_od_f = filter_based_on_requirements(class_df_od_f, "model_size", max_footprint, minimum=False)
    if class_df_od_f.empty:
        assert False, f"Model size requirement not fulfilled"
    if "remnet" in model:
        class_df_od_f = estimate_runtime_remnet(class_df_od_f)
    else:
        class_df_od_f = calculate_FLOPs(model, class_df_od_f)
        class_df_od_f = estimate_runtime_with_FLOPs(class_df_od_f, requirements["FLOPS"])
    class_df_od_f = filter_based_on_requirements(class_df_od_f, "estimated_runtime", max_runtime, minimum=False)     
    if class_df_od_f.empty:
        # assert False, f"Runtime requirement not fulfilled"
        log.info(f"Runtime requirement not fulfilled")
    log.info("Best classification candidates")
    log.info(f"""{class_df_od_f[["score_accuracy"] + params]}""")   
    
    reg_df_od_f = reg_df_od
    reg_df_od_f = filter_based_on_requirements(reg_df_od_f, "model_size", max_footprint, minimum=False)
    if reg_df_od_f.empty:
        assert False, f"Model size requirement not fulfilled"
    if "remnet" in model:
        reg_df_od_f = estimate_runtime_remnet(reg_df_od_f)
    else:
        reg_df_od_f = calculate_FLOPs(model, reg_df_od_f)
        reg_df_od_f = estimate_runtime_with_FLOPs(reg_df_od_f, requirements["FLOPS"])
    reg_df_od_f = filter_based_on_requirements(reg_df_od_f, "estimated_runtime", max_runtime, minimum=False)  
    if reg_df_od_f.empty:
        # assert False, f"Runtime requirement not fulfilled"
        log.info(f"Runtime requirement not fulfilled")
    log.info("Best error correction candidates")
    log.info(f"""{reg_df_od_f[["score_r2"] + params]}""")   
    log.info("")
    
    if model == "remnet_mo":
        datapoints_class = class_df_od_f[[objective_class, objective_reg]].values.transpose()
        datapoints_reg = reg_df_od_f[[objective_class, objective_reg]].values.transpose()
    else:
        datapoints_class = class_df_od_f[[objective_class]].values.transpose()
        datapoints_reg = reg_df_od_f[[objective_reg]].values.transpose()

    optimal_datapoints_class, indices_class = pareto_front(datapoints_class)
    class_df_od_best_candidates = class_df_od_f.iloc[list(indices_class)]
    optimal_datapoints_reg, indices_reg = pareto_front(datapoints_reg)
    reg_df_od_best_candidates = reg_df_od_f.iloc[list(indices_reg)]

    weights = np.array([0.5, 0.5])
    decomp = ASF()
    
    index_class = index_reg = []
    if class_df_od_f.empty == False:
        index_class = decomp(optimal_datapoints_class.transpose(), weights).argmin()
    if reg_df_od_f.empty == False:
        index_reg = decomp(optimal_datapoints_reg.transpose(), weights).argmin()
    
    class_df_od_best = class_df_od_best_candidates.iloc[index_class]
    reg_df_od_best = reg_df_od_best_candidates.iloc[index_reg]
    
    log.info("Best error correction model")
    log.info(f"""{class_df_od_best[["score_accuracy"] + params]}""")   
    log.info("")
    
    log.info("Best error correction model")
    log.info(f"""{reg_df_od_best[["score_r2"] + params]}""")   
    log.info("")
    class_df_od_best["model"] = model
    reg_df_od_best["model"] = model
    if class_df_od_f.empty == False:
        df_res.append(class_df_od_best)
    if reg_df_od_f.empty == False:
        df_res.append(reg_df_od_best)
    
#%% Use Achievement Scalarization Function (ASF) to find optimal solution. Also works for multi-output ML models where accuracy and r2 score need to be evaluated together.
classification = []
regression = []
for i in range(int(len(df_res) / 2)):
    log.info(f"########### {df_res[2*i].model} ############")
    log.info("Best error correction model")
    log.info(f"""{df_res[2*i][["score_accuracy"]]}""")   
    log.info("")
    classification.append(df_res[2*i][["score_accuracy"]])
    
    log.info("Best error correction model")
    log.info(f"""{df_res[(2*i)+1][["score_r2"]]}""")   
    log.info("")
    regression.append(df_res[(2*i) + 1][["score_r2"]])

datapoints = np.hstack((classification, regression)).transpose()
optimal_datapoints, indices = pareto_front(datapoints)

if len(indices) > 1:
    weights = np.array([0.5, 0.5])
    decomp = ASF()
    I = decomp(optimal_datapoints.transpose(), weights).argmin()
    print("Best regarding decomposition: Point %s - %s" % (I, optimal_datapoints.transpose()[I]))
else:
    I = indices[0] 

class_df_od_best = pd.DataFrame([df_res[2* I]])
reg_df_od_best = pd.DataFrame([df_res[2*I + 1]])

    
#%% Save best suitable model config
def save_best_suitable_config_as_json(data, output_path, model, interesting_params):
    string = data[interesting_params].to_json(f"{output_path}/model_selection_{model}_best_suitable_config.json", orient="records")
 
model = class_df_od_best.model.values[0]
save_best_suitable_config_as_json(class_df_od_best, output_path, f"{model}_class", grid_search_config[model])
save_best_suitable_config_as_json(reg_df_od_best, output_path, f"{model}_reg", grid_search_config[model])
 