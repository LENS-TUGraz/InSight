#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 13:59:56 2022

@author: michi
"""
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_data_set_distribution(_df):
    
    fig, axes = plt.subplots(1,2)
    
    ax = axes[0]
    if "real_rng" in _df.columns:
        dip = _df
        dip['dd'] = _df['real_rng'].apply(lambda a: int(a/2)*2)
        data_dist = dip.groupby(["dd","case"]).size().reset_index()
        sns.barplot(x="dd",y=0,data=data_dist, hue="case", ax = ax)
        ax = plt.gca()
        ax.set_xlabel("Distance")
        ax.set_ylabel("Number of samples")
        
        ax = axes[1]
        dip['dd'] = _df['real_rng'].apply(lambda a: int(a/2)*2)
        data_dist = dip.groupby(["dd","NLOS"]).size().reset_index()
        sns.barplot(x="dd",y=0,data=data_dist, hue="NLOS", ax = ax)
        ax = plt.gca()
        ax.set_xlabel("Distance")
        ax.set_ylabel("Number of samples")
        
        plt.pause(0.1)
        

        # n_splits = 5
        
        
        
            # if "rng_e" in dfc.columns:
            #     dip['dd'] = dfc['real_rng'].apply(lambda a: int(a/2)*2)
            #     data_dist = dip.groupby(["dd","case"]).size().reset_index()
            #     sns.barplot(x="dd",y=0,data=data_dist, hue="case", ax = ax)
            #     ax = plt.gca()
            #     ax.set_xlabel("Distance")
            #     ax.set_ylabel("Number of samples")
            # assert(False)
            
                    
                # kf = KFold(n_splits=training_configuration['kfolds'],shuffle=True)
                # for it, (train, test) in enumerate(kf.split(dfc)):
                # dataset_split = "k_fold" # "k_fold", "single"
                # data_fname, dclasses = groupby_fname_and_get_dclasses(dfc)
                # params = {"n_splits": n_splits, "data_to_split": data_fname, "stratification_base": dclasses}
                # split_generator = choose_dataset_split_method(dataset_split, params, shuffle=False)
                # for it, (train, test) in enumerate(split_generator): # 
