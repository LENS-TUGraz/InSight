# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 19:02:02 2023

@author: turtle9
"""



#%% Pareto
import oapackage

def pareto_front(datapoints):    
    pareto=oapackage.ParetoDoubleLong()
    for ii in range(0, datapoints.shape[1]):
        data = tuple([datapoints[i, ii] for i in range(datapoints.shape[0])])
        w=oapackage.doubleVector(data)
        pareto.addvalue(w, ii)
    pareto.show(verbose=0)
    lst=pareto.allindices() # the indices of the Pareto optimal designs
    return datapoints[:,lst], lst
    

import tensorflow as tf
    
import scipy.interpolate    
def estimate_runtime_remnet(df):
    y = [0.003, 0.008, 0.062]
    # x = [7239, 86547, 689511] # flops
    x = [7, 12, 28] # model size
    y_interp = scipy.interpolate.interp1d(x, y)
    # df["estimated_runtime"] = y_interp(df.flops)
    df["estimated_runtime"] = y_interp(df.model_size)
    return df
    
    
def estimate_runtime_with_FLOPs(reg_df_od_f, FLOPS):
    reg_df_od_f["estimated_runtime"] = reg_df_od_f.flops / FLOPS
    return reg_df_od_f


def filter_based_on_requirements(data, objective, requirement, minimum=False):
    if requirement != -1:
        if minimum:
            data = data[data[objective] > requirement]
        else:
            data = data[data[objective] < requirement]
    return data
 