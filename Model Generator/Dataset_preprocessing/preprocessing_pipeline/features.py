#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 14:54:45 2022

@author: michi
"""
import logging
log = logging.getLogger('FEATC')
if not log.hasHandlers():
    log.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(name)s:%(levelname)s:%(message)s')
    ch.setFormatter(formatter)
    log.addHandler(ch)
    
import numpy as np
import pandas as pd
from fastcore.transform import *

def calc_energy(d):
    return np.sum(np.power(d['cir'],2))

def calc_max(d):
    return np.max(d['cir'])

def calc_std(d):
    return np.std(d['cir_bak'][-10:])

def calc_th_tL(d):
    return 6*d['std_noise']

def calc_th_tH(d):
    return 0.6*d['f2']

def calc_rise_time(d):
    c = np.array(d['cir'])
    try:
        iL = np.argwhere(c>d['thL'])[0][0]
        iH = np.argwhere(c>d['thH'])[0][0]
    except:
        # print(d)
        # print(np.argwhere(c>d['thL']))
        # print(np.argwhere(c>d['thH']))
        iH = 0
        iL = 0
    return (iH - iL)

def calc_fp_tmed(d):
    try:
        c = np.array(d['cir'])
        iL = np.argwhere(c>d['thL'])[0][0]
        iH = d['f4']
        return (iH - iL)
    except:
        print("Error: FP TMED calculation failed")
        return np.nan
    

def calc_tau_med(d):
    rs = np.power(d['cir'],2)
    phi = rs/np.sum(rs)
    t = np.arange(0, np.size(phi))
    tau_med = np.sum(t*phi)

    tau_rms = np.sqrt(np.sum(np.power(t- tau_med,2)*phi))
    return tau_med, tau_rms

def calc_kurtosis(d):
    T = np.size(d['cir'])
    mu_r = np.mean(d['cir'])
    sig_r = np.var(d['cir'])
    k = np.sum(np.power(np.abs(d['cir']) -  mu_r,4))
    k = k / (np.power(sig_r,2)*T)
    return k

def calc_pre_noise(d):
    return np.std(d.cir[d.fp_idx-10:d.fp_idx-2])

def calc_FPPL(a):
    #TODO: Different A for different PRF
    A = 121.74
    return 10*np.log10((a.fp_amp1**2 + a.fp_amp2**2 + a.fp_amp3**2)/a.pacc_cnt**2) - A

def calc_LOS(a):
    d = a.rssi - a.FPPL
    if d < 6:
        return 1
    elif d > 10:
        return 0
    else:
        t = 1 - (d -6)/4
        return t
# _df = data

def calc_features(_df, cir_start, cir_end):
    '''
    Calculates the features of the supplied measurements
    Parameters
    ----------
    _df : DataFrame
    Returns    'features':['f1','f2','f3','f4','f5','f6', 'f8','f11','f7'], #,'f11'], #, ,"f14" #,'f7' 'f7',

    -------
    _df : DataFrame
    '''
    log.warning("n_measurments is ignored because it destroys the original dataset")
    warning_state = pd.options.mode.chained_assignment
    pd.options.mode.chained_assignment = None
    log.warning("Chained assignmnet warning switched off")
    log.info("CIR length is reset after feature calculation")
    log.info("Set correct CIR window and FP index")
    _df['cir_bak'] = _df['cir']
    _df['cir'] = _df.apply(lambda a: a.cir[cir_start:cir_end],axis=1) 
    log.info(f"Feature Calculation CIR length is {cir_end - cir_start}")
    _df['std_noise'] = _df.apply(lambda x: calc_std(x), axis=1)
    _df['f1'] = _df.apply(lambda x: calc_energy(x),axis=1)
    _df['f2'] = _df.apply(lambda x: calc_max(x),axis=1)
    _df['thL'] = _df.apply(lambda x: calc_th_tL(x),axis=1)
    _df['thH'] = _df.apply(lambda x: calc_th_tH(x),axis=1)
    _df['f3'] = _df.apply(lambda x: calc_rise_time(x),axis=1)
    _df[['f4','f5']] = _df.apply(lambda x: calc_tau_med(x),axis=1, result_type="expand")
    _df['f6'] = _df.apply(lambda x: calc_kurtosis(x),axis=1)
    if "rng" in _df.columns:
        _df['f7'] = _df['rng']
    _df['f8'] = _df.apply(lambda x: calc_fp_tmed(x),axis=1) 
    _df['f11'] = _df.apply(lambda x: calc_pre_noise(x),axis=1)
    _df['cir'] = _df['cir_bak']
    log.info(f"Restoring CIR length after feature calculation: {len(_df.iloc[0].cir_bak)}")
    _df.drop("cir_bak",inplace=True,axis=1)
    pd.options.mode.chained_assignment = warning_state
    return _df

class Calculate_Features(Transform):
    def setup(self, config, train_setup): 
        self.cir_start = config['cir_start']
        self.cir_end = config['cir_end']
    def encodes(self, _df): 
        return calc_features(_df, self.cir_start, self.cir_end)