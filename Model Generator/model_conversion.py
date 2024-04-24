# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 13:59:56 2023

@author: turtle9
"""

""" MODEL CONVERSION """
import logging

log = logging.getLogger('MLCON')
if not log.hasHandlers():
    log.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(name)s:%(levelname)s:%(message)s')
    ch.setFormatter(formatter)
    log.addHandler(ch)
    
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, r2_score
from Hyper_parameter_optimization.data_split import split_dataframe,split_test_val
import pandas as pd
from fastcore.transform import Pipeline
import tensorflow as tf
import numpy as np
from sklearn.utils.class_weight import compute_sample_weight
from Model_conversion.model_conversion_helper import *
from Hyper_parameter_optimization.training_configurations import grid_search_metadata
import yaml

seed = 100
np.random.seed(seed)
tf.keras.utils.set_random_seed(seed)

    
config = {}
with open('./Model_conversion/model_conversion_config.yaml') as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)
    
hpo_config = {}
with open('./Hyper_parameter_optimization/hyper_parameter_optimization_config.yaml') as f:
    hpo_config = yaml.load(f, Loader=yaml.SafeLoader)
    
dp_config = {}
with open("Dataset_preprocessing/dataset_input_config.yaml") as f:
    dp_config = yaml.load(f, Loader=yaml.SafeLoader)

fp_pos = config["fp_pos"]
model_to_run = config["model_to_run"] 
training_configuration = hpo_config["training_configuration"]#
test_model = config["test_model"] 
use_kfold = config["use_kfold"]
use_validation_set = config["use_validation_set"]

preprocessed_data_path = "Datasets/preprocessed_datasets"
best_suitable_config_path = "Model_selection/model_selection_output"
training_data_path = "train_data.pkl"
testing_data_path = "test_data.pkl"
output_path = "Model_conversion/model_conversion_output"

class_mlmod = None
reg_mlmod = None
mo_mlmod = None


""" TRAIN MODEL """
log.info("")
log.info("Training model with best suitable hyper-parameters")
log.info("Getting best suitable config from file")          
if "class" in model_to_run["mltype"]:
    class_best_config = pd.read_json(f"""{best_suitable_config_path}/model_selection_{model_to_run["model"]}_class_best_suitable_config.json""", orient="records").iloc[0].to_dict()
if "reg" in model_to_run["mltype"]:
    reg_best_config = pd.read_json(f"""{best_suitable_config_path}/model_selection_{model_to_run["model"]}_reg_best_suitable_config.json""", orient="records").iloc[0].to_dict()
if "mo" in model_to_run["mltype"]:
    mo_best_config = pd.read_json(f"""{best_suitable_config_path}/model_selection_{model_to_run["model"]}_class_best_suitable_config.json""", orient="records").iloc[0].to_dict()

def train_model(train_data, mlmod, best_config, model, mltype):
    """ MIX CONFIG """
    best_config["early_stopping"] = training_configuration["early_stopping"]
    mlmod = grid_search_metadata[model][mltype]["method"](**grid_search_metadata[model][mltype]["fixed_config"], **best_config)
    
    from Dataset_preprocessing.preprocessing_pipeline.preprocessing import Filter_Conditions,  KF_Grouping, Filter_Resampling, ScaleRNGEMinMax, Filter_Environments
    pipeline_filter = Pipeline([
        Filter_Conditions,
        Filter_Resampling,
        ScaleRNGEMinMax,
        KF_Grouping
        ])
    pipeline_filter.setup(training_configuration, train_setup=False)
    

    dff1 = pipeline_filter(train_data)
    dff2 = pipeline_filter(train_data) # not used, passed through to get indexes
    if use_kfold:
        split_iterator = split_dataframe(dff1, training_configuration['kfolds'], setup_name="setup_id", train_size=0.8, validation=use_validation_set, seed=seed)
    else:
        split_iterator = split_test_val(dff1,dff2,train_size=0.8, seed=seed)
        
    for it,(train, test, val) in enumerate(split_iterator):
        dff_train = dff1.loc[train]
        dff_val = dff1.loc[val] 
        dff_test = dff2.loc[test]
                    
        mlmod.fit(dff_train, dff_val) 
    return mlmod  

training_data = pd.read_pickle(f"{preprocessed_data_path}/{training_data_path}")[1]
if "class" in model_to_run["mltype"]:
    log.info(f"""Training {model_to_run["model"]} Classifier""")
    class_mlmod = train_model(training_data, class_mlmod, class_best_config, model_to_run["model"], "class")

if "reg" in model_to_run["mltype"]:
    log.info(f"""Training {model_to_run["model"]} Regressor""")
    reg_mlmod = train_model(training_data, reg_mlmod, reg_best_config, model_to_run["model"], "reg")

if "mo" in model_to_run["mltype"]:
    log.info(f"""Training {model_to_run["model"]} Multi-output""")
    mo_mlmod = train_model(training_data, mo_mlmod, mo_best_config, model_to_run["model"], "mo")


""" TEST MODEL """
if test_model:
    log.info("")
    log.info("Testing performance of model")
    if testing_data_path:
        test_data = pd.read_pickle(f"{preprocessed_data_path}/{testing_data_path}")[1]
    if class_mlmod:
        y_class_hat, y_class = class_mlmod.predict(test_data)
        weights = compute_sample_weight(class_weight="balanced",y=y_class)
        log.info(f"F1 score: {f1_score(y_class,y_class_hat,sample_weight=weights)}")
        log.info(f"Accuracy: {accuracy_score(y_class,y_class_hat,sample_weight=weights)}")
        log.info(f"Precision: {precision_score(y_class,y_class_hat,sample_weight=weights)}")
        log.info(f"Recall: {recall_score(y_class,y_class_hat,sample_weight=weights)}")
    if reg_mlmod:
        y_reg_hat_scaled, y_reg_scaled, y_reg_hat = reg_mlmod.predict(test_data)
        log.info(f"R2 score: {r2_score(y_reg_scaled, y_reg_hat_scaled)}")
    if mo_mlmod:
        y_reg_hat_scaled, y_reg_scaled, y_reg_hat, y_class_hat, y_class = mo_mlmod.predict(test_data)
        weights = compute_sample_weight(class_weight="balanced",y=y_class)
        log.info(f"F1 score: {f1_score(y_class,y_class_hat,sample_weight=weights)}")
        log.info(f"Accuracy: {accuracy_score(y_class,y_class_hat,sample_weight=weights)}")
        log.info(f"Precision: {precision_score(y_class,y_class_hat,sample_weight=weights)}")
        log.info(f"Recall: {recall_score(y_class,y_class_hat,sample_weight=weights)}")
        log.info(f"R2 score: {r2_score(y_reg_scaled, y_reg_hat_scaled)}")
#%%
""" CONVERT MODEL """
log.info("")
log.info("Converting model")
if class_mlmod:
    if "remnet" == model_to_run["model"]:
        class_mlmod.save(f"""{output_path}/{model_to_run["model"]}_class""")
        class_mlmod.convert_saved_model_to_tflite(f"""{output_path}""", f"""{model_to_run["model"]}_class""", quantization="full_integer")
    else:
        class_mlmod.convert_to_lite(f"""{output_path}/{model_to_run["model"]}_class.h""", 1, add_scaling_values=False)
if reg_mlmod:
    if "remnet" == model_to_run["model"]:
        reg_mlmod.save(f"""{output_path}/{model_to_run["model"]}_reg""")
        reg_mlmod.convert_saved_model_to_tflite(f"""{output_path}""", f"""{model_to_run["model"]}_reg""", quantization="full_integer")
    else:
        reg_mlmod.convert_to_lite(f"""{output_path}/{model_to_run["model"]}_reg.h""", add_scaling_values=False)
if mo_mlmod:
    mo_mlmod.save(f"""{output_path}/{model_to_run["model"]}_mo""")
    mo_mlmod.convert_saved_model_to_tflite(f"""{output_path}""", f"""{model_to_run["model"]}_mo""", quantization="full_integer")

""" CREATE CIR CONFIG HEADER"""
log.info("Creating CIR config header .h file")

cir_fp_pos = fp_pos
class_cir_start = class_cir_end = reg_cir_start = reg_cir_end = class_scale_len = reg_scale_len = -1
if class_mlmod:
    class_cir_start = int(class_mlmod.params["cir_start"])
    class_cir_end = int(class_mlmod.params["cir_end"])
    class_scale_len = len(class_mlmod.tra.x_min)
    if "remnet" == model_to_run["model"]:
        class_scale_len = int(class_mlmod.tra.x_min)
if reg_mlmod:
    reg_cir_start = int(reg_mlmod.params["cir_start"])
    reg_cir_end = int(reg_mlmod.params["cir_end"])
    reg_scale_len = len(reg_mlmod.tra.x_min)
    if "remnet" == model_to_run["model"]:
        reg_scale_len = int(reg_mlmod.tra.x_min)
if mo_mlmod:
    class_cir_start = int(mo_mlmod.params["cir_start"])
    class_cir_end = int(mo_mlmod.params["cir_end"])
    class_scale_len = int(mo_mlmod.tra.x_min)
    reg_cir_start = int(mo_mlmod.params["cir_start"])
    reg_cir_end = int(mo_mlmod.params["cir_end"])
    reg_scale_len = int(mo_mlmod.tra.x_min)

define_out = generate_cir_config_macro()
define_out += generate_cir_config_header(class_cir_start, class_cir_end, cir_fp_pos, reg_cir_start, reg_cir_end)
define_out += generate_scaling_parameter_config_header(class_scale_len, reg_scale_len)
define_out += """
#endif
"""
with open(f"""{output_path}/{model_to_run["model"]}_model_config_output.h""","w") as f:
    f.write(define_out)
    
    
""" CREATE CIR CONFIG HEADER CODE"""
log.info("Creating CIR config header .c file")
class_scale_x_min = class_scale_x_max = reg_scale_x_min = reg_scale_x_max = [-1]
reg_scale_y_min = reg_scale_y_max = -1
if class_mlmod:
    class_scale_x_min = class_mlmod.tra.x_min
    class_scale_x_max = class_mlmod.tra.x_max
if reg_mlmod:
    reg_scale_x_min = reg_mlmod.tra.x_min
    reg_scale_x_max = reg_mlmod.tra.x_max
    reg_scale_y_min = reg_mlmod.tra.y_min
    reg_scale_y_max = reg_mlmod.tra.y_max
if mo_mlmod:
    class_scale_x_min = mo_mlmod.tra.x_min
    class_scale_x_max = mo_mlmod.tra.x_max
    reg_scale_x_min = mo_mlmod.tra.x_min
    reg_scale_x_max = mo_mlmod.tra.x_max
    reg_scale_y_min = mo_mlmod.tra.y_min
    reg_scale_y_max = mo_mlmod.tra.y_max
    
if "remnet" == model_to_run["model"]:
    define_out = generate_scaling_parameter_code_mo(class_scale_x_min, class_scale_x_max, reg_scale_x_min, reg_scale_x_max, reg_scale_y_min, reg_scale_y_max)
else:
    define_out = generate_scaling_parameter_code(class_scale_x_min, class_scale_x_max, reg_scale_x_min, reg_scale_x_max, reg_scale_y_min, reg_scale_y_max)
with open(f"""{output_path}/{model_to_run["model"]}_model_config_output.c""","w") as f:
    f.write(define_out)

log.info("")
log.info("DONE")





