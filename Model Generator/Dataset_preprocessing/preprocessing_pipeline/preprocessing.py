import logging
log = logging.getLogger('PREPR')
if not log.hasHandlers():
    log.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(name)s:%(levelname)s:%(message)s')
    ch.setFormatter(formatter)
    log.addHandler(ch)

from fastcore.transform import *
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Filter_Conditions(Transform):
    def setup(self, items, train_setup): self.conditions = items['conditions']
    def encodes(self, _df): 
        log.info(f"Filter Conditions {self.conditions}")
        return _df[_df.labels.isin(self.conditions)]
    
class Filter_Environments(Transform):
    def setup(self, items, train_setup): self.conditions = items['environments']
    def encodes(self, _df): 
        log.info(f"Filter Environments {self.conditions}")
        return _df[_df.env1.isin(self.conditions)|_df.env2.isin(self.conditions)]
    
class KF_Grouping(Transform):
    def setup(self, items, train_setup): self.kfgroups = items['kfgroups']
    def encodes(self, _df):
        log.info(f"Generating K-Fold groupby label: {self.kfgroups}")
        _df['kfgroup'] = _df[self.kfgroups].astype(str).agg("_".join,axis=1) 
        return _df
    
class ScaleCIRMinMax(Transform):
    def encodes(self, _df):
        _df['kfgroup'] = _df[self.kfgroups].astype(str).agg("_".join,axis=1) 
        return _df

class ScaleEachCIR(Transform):
    def encodes(self, _df):
        _df['cir'] = _df['cir'].apply(lambda a: a.cir/np.max(a.cir),axis=1)

        return _df
    
class ScaleRNGEMinMax(Transform):
    def encodes(self, _df):
        if "rng_e" in _df.columns:
            scaler = MinMaxScaler()
            weights = scaler.fit_transform(_df.rng_e.abs().values.reshape(-1,1)).flatten()
            _df['rng_e_scaled'] = weights
        else:
            log.warning("[ScaleRNGEMinMax]: Warning: No ranging error for WA scaling avaliable")
            _df['rng_e_scaled'] = 0
        return _df
''' 
    Pipeline prior KF Split
'''
class Filter_Resampling(Transform):
    def setup(self, items, train_setup): 
        self.resample = items['resample']
        self.resample_tag = items['resample_tag']
        
    def encodes(self, _df):
        log.info(f"Resampling samples")
        if self.resample:
            gr = _df.groupby("setup_id").first().groupby(self.resample_tag)
            s = gr.sample(gr.size().min())
            _df = _df[_df.setup_id.isin(s.index)]
        return _df
    
class Remove_Greater(Transform):
    def setup(self, max_rng, train_setup): 
        self.max_rng = max_rng['max_rng']
        self.min_rng = max_rng['min_rng']
    def encodes(self, _df): 
        log.info(f"Remove real range greater {self.max_rng}")
        if "real_rng" not in _df.columns:
            log.warning("[Remove_Greater] real_rng not in column removing skipped")
            return _df
        return _df[(_df.real_rng<self.max_rng)&(_df.real_rng>self.min_rng)]
    
class Remove_RNGE_Greater(Transform):
    def setup(self, max_rng, train_setup): 
        self.max_rng = max_rng['max_rng_e']
        self.min_rng = max_rng['min_rng_e']
    def encodes(self, _df): 
        return _df[(_df.rng_e<self.max_rng)&(_df.rng_e>self.min_rng)]
    
    
''' UWB Bias Correction '''
def path_loss(Pt, G, fc, R):
    """
    * @fn uwb_rng_path_loss(float Pt, float G, float fc, float R)
    * @brief calculate rng path loss using range parameters and return signal level.
    * and allocated PANIDs and SLOTIDs.
    *
    * @param Pt      Transmit power in dBm.
    * @param G       Antenna Gain in dB.
    * @param Fc      Centre frequency in Hz.
    * @param R       Range in meters.
    *
    * @return Pr received signal level dBm
    """
    Pr = Pt + 2 * G + 20 * np.log10(299792458/1.000293) - 20 * np.log10(4 * np.pi * fc * R)
    return Pr



def bias_correction(Pr):
    """
    * @fn uwb_rng_bias_correction(struct uwb_dev * inst, float Pr)
    * @brief API for bias correction polynomial.
    *
    * @param pr     Variable that calculates range path loss.
    *
    * @return Bias value
    """
    p = [1.404476e-03, 3.208478e-01, 2.349322e+01, 5.470342e+02]
    bias = np.polyval(p, Pr)
            
    return bias / 100 # in m

class BiasCorrection(Transform):
    def setup(self, items, train_setup): 
        self.bias_correction = items['bias_correction']
        
    def encodes(self, _df):
        log.info(f"Bias Correction {self.bias_correction}")
        if self.bias_correction == "none":
            if ("rng" in _df.columns) and ("real_rng" in _df.columns):
                _df['rng_e'] = _df['rng'] - _df['real_rng']
            else:
                log.info("Either rng, or real_rng not in dataframe. Skipping ranging error calculation")
            return _df
        elif self.bias_correction == "given":
            # _df['bias_calculated'] = _df.apply(lambda a: a.rng - bias_correction(path_loss(Pt,G,fc,a.rng)),axis=1)
            _df['rng'] = _df['rng'] - _df['bias']
            _df['rng_e'] = _df['rng'] - _df['real_rng']
            return _df
        elif self.bias_correction == "poly":
            Pt = -14.3
            G = 1
            fc = 6489.4e6
            _df['bias_calculated'] = _df.apply(lambda a: bias_correction(path_loss(Pt,G,fc,a.rng)),axis=1)
            _df['rng_wo_bias_calculated'] = _df['rng'] - _df['bias_calculated']
            _df['rng_e'] = _df['rng_wo_bias_calculated'] - _df['real_rng']
            return _df
        else:
            assert(False)
""""""


""" Dataset cleaning """
class Data_Cleaning(Transform):
    def setup(self, items, train_setup):
        self.filter_negative_values_column = items["filter_negative_values_column"]
        
        self.outlier_column = items["outlier_column"]
        self.groupby_columns = items["groupby_columns"]
        self.verbose = items["filter_outlier_verbose"]
        
    def encodes(self, _df):
        log.info("Data Cleaning")
        if self.filter_negative_values_column != None:
            _df = filter_negative_values(_df, self.filter_negative_values_column)
        if "rng" not in _df.columns:
            log.warning(f"[Data_Cleaning] Column: {self.outlier_column} not in dataframe skipping data cleaning")
            
            return _df
        
        if self.outlier_column != None:
            if "outlier" in _df:
                log.info(f"{self.outlier_column} outliers already filtered. Not filtering again to not change the distribution")
                return _df
            
            groups = _df.groupby(self.groupby_columns)
            warning_state = pd.options.mode.chained_assignment
            pd.options.mode.chained_assignment = None
            log.warning("Chained assignmnet warning switched off")
            for f, g in groups:
                _df.loc[(g[self.groupby_columns] == f).index, "outlier"] = find_outliers_IQR(g, self.outlier_column, verbose=self.verbose)
        
            _df= _df[_df.outlier == False]
            pd.options.mode.chained_assignment = warning_state
        return _df
        
    
def filter_negative_values(df, column):
    if (df[column] < 0).sum():
        log.info(f"Found {(df[column] < 0).sum()} negative ranges. Discarding {df[df[column] < 0][column].values}...")
    else:
        log.info("No negative ranges found")
    return df[df[column] > 0]


def find_outliers_IQR(df, column, verbose=False):
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    IQR = q3 - q1
    outliers = ((df[column] < (q1 - 5 * IQR)) | (df[column] > (q3 + 5 * IQR)))
    if verbose and outliers.sum():
        log.info(f"Dropping {column} outliers: {outliers.sum()} of {len(df)}! Mean={round(df[column].mean(), 2)} STD={round(df[column].std(), 2)} Ranges={df[outliers][column].values}")
        
    if verbose and outliers.sum():
        """ Plot {column} before and after filtering outliers """
        fig, ax = plt.subplots(2, 1)
        fig.suptitle(f"{column.upper()} Outliers")
        ax[0].plot(df[column], c="blue", label="Unfiltered")
        ax[0].legend()
        data_clean = df[~outliers]
        ax[1].plot(data_clean[column], c="orange", label="Filtered")
        ax[1].legend()
        plt.show()
    return outliers

""""""