# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 17:37:31 2018

Titanic Competition in Kaggle
Dataset path : E:\Data_and_Script\Python_Script\titanic\train.csv'
Data from : Kaggle.com

Python 3.6.3 
GUI : Spyder from Anaconda 5.0.1
OS : windows 10 v1709 64 bit

P1 : EDA
P2 : Feature Engineering
P3 : Model and Ensemble

@author: Haby
"""
# ----------------------------------------------------------------- #
# ----------------------------- EDA ------------------------------- #
# ----------------------------------------------------------------- #

# import Package

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
import warnings
warnings.filterwarnings('ignore')

# import data
train = pd.read_csv(r'E:\Data_and_Script\Python_Script\titanic\train.csv')
test = pd.read_csv(r'E:\Data_and_Script\Python_Script\titanic\test.csv')

# check NAs
train.isnull().sum()
