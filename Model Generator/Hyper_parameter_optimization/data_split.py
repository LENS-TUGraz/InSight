#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 14:01:33 2022

@author: michi
"""
from sklearn.model_selection import GroupKFold, KFold,GroupShuffleSplit,StratifiedGroupKFold, ShuffleSplit

def split_dataframe(_df, folds, setup_name="setup_id", train_size=None, validation=False, seed=None):
    
    # StratifiedGroupKFold
    kf = GroupKFold(n_splits=folds)
    si = _df.groupby(setup_name).first()
    for it, (_si_train, si_test) in enumerate(kf.split(si,groups=si.kfgroup)): #,y=si.NLOS
        test_setupid = si.iloc[si_test].index
        test = _df[_df[setup_name].isin(test_setupid)].index
        
        if validation == True:
            gss = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=seed)
            si2 = si.iloc[_si_train]
            si_train, si_val = next(gss.split(si2, si2.NLOS, groups=si2.kfgroup))
        
            train_setupid = si2.iloc[si_train].index
            val_setupid = si2.iloc[si_val].index
    
            train = _df[_df.setup_id.isin(train_setupid)].index
            val = _df[_df.setup_id.isin(val_setupid)].index
        
            yield train, test, val
        else:
            train_setupid = si.iloc[_si_train].index
            train = _df[_df[setup_name].isin(train_setupid)].index

            
            yield train, test, []
            
def split_test_val(_dff1,_dff2, train_size=None, seed=None):
        ss = ShuffleSplit(n_splits=1, train_size=train_size, random_state=seed)  
        for train,val in ss.split(_dff1):
            train = _dff1.iloc[train]
            val = _dff1.iloc[val]
            yield train.index, _dff2.index, val.index, 
# for train, test, val in split_dataframe(dfc, 3, "setup_id", validation=0.2):
    
#     assert(set(dfc.loc[train].setup_id.unique()).isdisjoint(set(dfc.loc[test].setup_id.unique())))
#     assert(set(dfc.loc[train].setup_id.unique()).isdisjoint(set(dfc.loc[val].setup_id.unique())))
    
#     plot_data_set_distribution(dfc.loc[test])