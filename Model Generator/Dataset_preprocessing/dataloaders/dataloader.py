#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 14:38:56 2022

@author: michi
"""
import logging
log = logging.getLogger("DATAL")
if not log.hasHandlers():
    log.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(name)s:%(levelname)s:%(message)s')
    ch.setFormatter(formatter)
    log.addHandler(ch)
    
import pandas as pd
import numpy as np


def load_data_stocker_unf(fname):
    data = pd.read_feather(fname)
    data = data.reset_index().rename(columns={"index":"original_index"})
    data = data.rename(columns={'location_from':"env1","location_to":"env2"})
    data['rng_e'] = data['rng'] -data['real_rng']
    data['NLOS'] = 0
    data.loc[data.labels=="wlos",'NLOS'] = 1
    data.loc[data.labels=="nlos",'NLOS'] = 1    

    # Roll CIR to move first path to fixed position
    ROLL_CIR = 20
    data['cir'] = data.apply(lambda a: np.roll(a.cir,-int(a.fp_idx)+ROLL_CIR),axis=1)
    data['fp_idx'] = data.apply(lambda a: ROLL_CIR,axis=1)
    
    # CIR is already normalized
    log.warning("load_data_stocker_unf: CIR is already normalized")
    data["std_noise"] = data.apply(lambda a: a.std_noise / a.pacc_cnt, axis=1)
    return data


""" Defines dictionary, keys (short dataset names) are put in dataset_input_config.yaml and define which datasets are loaded. """
dataset_db = {}
dataset_db['stocker_unf'] = {"name":"Stocker (III) Unfiltered", "file":f"""./Datasets/measurement_campaign/data_set_8a84_83a7_all.feather""", "loader":load_data_stocker_unf, "data_frame":None}

def load_dataset(dataset_id):
    df = dataset_db[dataset_id]["loader"](dataset_db[dataset_id]["file"])
    df["dataset_id"] = dataset_id
    return df
