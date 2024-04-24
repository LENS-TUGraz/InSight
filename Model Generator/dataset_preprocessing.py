# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 13:52:58 2023

@author: turtle9
"""

import logging

log = logging.getLogger('PREPR')
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
from fastcore.transform import Pipeline
from Dataset_preprocessing.dataloaders.dataloader import *
import seaborn as sns
import pickle as p
import yaml
from Dataset_preprocessing.preprocessing_pipeline.preprocessing import BiasCorrection, Remove_Greater, Remove_RNGE_Greater, Data_Cleaning
from Dataset_preprocessing.preprocessing_pipeline.features import Calculate_Features

seed = 100
np.random.seed(seed)
tf.keras.utils.set_random_seed(seed)

def load_yaml(file_name):
    file = {}
    with open(f"{file_name}.yaml") as f:
        file = yaml.load(f, Loader=yaml.SafeLoader)
    return file

config = load_yaml('./Dataset_preprocessing/dataset_input_config')
#%%
datasets_to_run = config["datasets_to_run"]

""" Load datasets """
ds_load_cnt = 0
for train_sets, test_sets in [list(datasets_to_run.values())]:
    datasets_to_load = train_sets + test_sets
    ''' Load training sets '''
    for ds in datasets_to_load:
        if dataset_db[ds]["data_frame"] is None:
            log.info(f"Loading {ds}")
            dataset_db[ds]["data_frame"] = load_dataset(ds)
            dataset_db[ds]["data_frame"]["setup_id"] = dataset_db[ds]["data_frame"].apply(lambda a: f"{ds_load_cnt}_{a.setup_id}",axis=1)
            dataset_db[ds]["data_frame"]["cir"] = dataset_db[ds]["data_frame"].apply(lambda a: a.cir[:400], axis=1)
            ds_load_cnt = ds_load_cnt + 1


""" Merge Datasets """
def merge_datasets(dataset_db, train):
    datasets_to_merge = [dataset_db[name]["data_frame"] for name in train] 
    name = " + ".join([dataset_db[name]["name"] for name in train])
    df = pd.concat(datasets_to_merge, ignore_index=True, axis=0)
    return (name, df)

data_sets = []
for train_sets, test_sets in [list(datasets_to_run.values())]:
    data_sets.append(merge_datasets(dataset_db, train_sets))
    data_sets.append(merge_datasets(dataset_db, test_sets))
    

#%%
preprocessing_config = config["pre_processing"]


''' 
    Pipeline for initial data pre-processing performing several tasks:
    - Calculate bias correction.
    - Remove outliers.
    - Confine range measurements to a maximum length.
    - Calculate features used for feature based methods.
'''
data_inital_processing = Pipeline([
    BiasCorrection,
    Remove_Greater,
    Remove_RNGE_Greater,
    Data_Cleaning,
    Calculate_Features,
    ])
data_inital_processing.setup(preprocessing_config, train_setup=False)

data_sets_inital_processed = []
for name, ds in data_sets:
    print(f"Preprocessing dataset {name}")
    dip = data_inital_processing(ds)
    data_sets_inital_processed.append((name,dip))

for idx, data in enumerate(data_sets_inital_processed):
    name = "train" if idx == 0 else "test"
    with open(f"./Datasets/preprocessed_datasets/{name}_data.pkl", "wb") as f:
        p.dump(data, f)
    