#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 13:19:15 2022

@author: michi
"""
from dataclasses import dataclass
from abc import abstractmethod


@dataclass
class MLMethodBase:
    name: str
    mltype: str
    params: tuple
    @abstractmethod
    def fit(self, Xtrain, ytrain, Xval=None, yval=None):
        pass
    
    @abstractmethod
    def predict(self, Xtest, ytest):
        pass