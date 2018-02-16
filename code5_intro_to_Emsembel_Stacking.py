# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 12:12:58 2018

Titanic Data
Code 5

@author: Kaggle kernel
"""

#%%
# import package and env variables
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import sklearn
import xgboost as xgb

import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls

import warnings
warnings.filterwarnings('ignore')

# 5 base model
from sklearn.ensemble import (RandomForestClassifier,AdaBoostClassifier,
                                     GradientBoostingClassifier,
                                     ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.cross_validation import KFold

#%%
# import data
train = pd.read_csv(r'E:\Data_and_Script\Python_Script\titanic\train.csv')
test = pd.read_csv(r'E:\Data_and_Script\Python_Script\titanic\test.csv')

# Store passengerID
PassengerId = test['PassengerId']

# head
train.head(3)

#%%
# ------------------- #
# FEATURE ENGINEERING #
# ------------------- #


full = [train,test]

# addin some new features
# len of name
train['Name_length'] = train['Name'].apply(len)
test['Name_length'] = test['Name'].apply(len)

# has cabin
train['Has_cabin'] = train['Cabin'].apply(lambda x : 0 if type(x) == float else 1)
test['Has_cabin'] = test['Cabin'].apply(lambda x : 0 if type(x) == float else 1)

#%%
# FamilySize
for d in full :
    d['FamilySize'] = d['Parch'] + d['SibSp'] + 1

# Create new feature ISALONE from familysize
for d in full :
    d['IsAlone'] = d['FamilySize'].apply(lambda x : 1 if x == 0 else 0)

# fill Nas in Embarked
for d in full :
    d['Embarked'] = d['Embarked'].fillna('S')

# fill Nas in Fare with median
for d in full :
    d['Fare'] = d['Fare'].fillna(d['Fare'].median())
# cut fare into 4 groups
for d in full :
    d['CategoricalFare'] = pd.qcut(d['Fare'],4)

#%%
# fill NAs of Age
for d in full :
    age_avg = d['Age'].mean()
    age_std = d['Age'].std()
    age_na_count = d['Age'].isnull().sum()
    age_na_rand_list = np.random.randint(age_avg - age_std,age_avg + age_std,
                                         size = age_na_count)
    d['Age'][np.isnan(d['Age'])] = age_na_rand_list
    d['Age'] = d['Age'].astype(int)
    d['CategoricalAge'] = pd.qcut(d['Age'],5)

# define function to extract from name
def get_title(name) :
    title_search = re.search(' ([A-Za-z]+)\.',name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""
# create new features Title
for d in full :
    d['Title'] = d['Name'].apply(get_title)

# group the non-common title with Rare
sns.countplot(x = 'Title',data = train,hue = 'Survived')
for d in full :
    d['Title'] = d['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don',
     'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    d['Title'] = d['Title'].replace(['Mlle','Ms'],'Miss')
    d['Title'] = d['Title'].replace('Mme','Mrs')

#%%
# plot
f,ax = plt.subplots(1,3,figsize = (10,5))
sns.countplot(x = 'Title',data = train,hue = 'Survived',ax = ax[0])
sns.countplot(x = 'Sex',data = train,hue = 'Survived',ax = ax[1])
sns.countplot(x = 'Embarked',data = train,hue = 'Survived',ax = ax[2])
#%%
# plot
f,ax = plt.subplots(2,1,figsize = (10,5))
sns.distplot(train['Age'],ax = ax[0])
sns.distplot(train['Fare'],ax = ax[1])

#%%
# define function for Fare and Age
def fare_map(x) :
    ''' Function for category Fare '''
    if x  <= 7.91 :
        return 0
    elif 7.91 < x <= 14.454 :
        return 1
    elif 14.454 < x <= 31 :
        return 2
    else :
        return 3
def age_map(x) :
    ''' Function for category Age '''
    if x <= 16 :
        return 0
    elif 16 < x <= 32 :
        return 1
    elif 32 < x <= 48 :
        return 2
    elif 48 < x <= 64 :
        return 3
    else :
        return 4
#%%
# convert catogetical to int
for d in full :
    # Sex to 1/0
    d['Sex'] = d['Sex'].map(lambda x : 0 if x == 'female' else 1).astype(int)
    # Title to 1-5
    title_mapping = {'Mr':1,'Miss':2,'Mrs':3,'Master':4,'Rare':5}
    d['Title'] = d['Title'].map(title_mapping).astype(int)
    # Embarked to 0-2
    d['Embarked'] = d['Embarked'].map({'S':0,'C':1,'Q':2}).astype(int)
    # Fare to 0-3
    d['Fare'] = d['Fare'].map(fare_map).astype(int)
    # Age to 0-4
    d['Age'] = d['Age'].map(age_map).astype(int)

#%%
# drop some features
drop_e = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp',
          'CategoricalAge', 'CategoricalFare']
train = train.drop(drop_e,axis = 1)
test = test.drop(drop_e,axis = 1)

#%%
    # VISUALIZATION

# Heatmap
colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0,
            square=True, cmap=colormap, linecolor='white', annot=True)

# PairPlot
g = sns.pairplot(data = train,hue = 'Survived',diag_kind = 'kde')
g.set(xticklabels = [])


#%%
    # Ensembling & Stacking models

# Some useful parameters which will come in handy later on
ntrain = train.shape[0]
ntest = test.shape[0]
SEED = 0 # for reproducibility
NFOLDS = 5 # set folds for out-of-fold prediction
kf = KFold(ntrain, n_folds= NFOLDS, random_state=SEED)

# Class to extend the Sklearn classifier

# new class sklearnhelper,with methods train,predict,fit,feature importances
class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)

    def fit(self,x,y):
        return self.clf.fit(x,y)

    def feature_importances(self,x,y):
        print(self.clf.fit(x,y).feature_importances_)

# Class to extend XGboost classifer

# return predicted value of train and mean predicted value of test
def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

#%%
   # Generating our Base First-Level Models¶

# Random Forest classifier
# Extra Trees classifier
# AdaBoost classifer
# Gradient Boosting classifer
# Support Vector Machine

# Put in our parameters for said classifiers
# Random Forest parameters
rf_params = {
    'n_jobs': -1,
    'n_estimators': 500,
     'warm_start': True,
     #'max_features': 0.2,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features' : 'sqrt',
    'verbose': 0
}
# Extra Trees Parameters
et_params = {
    'n_jobs': -1,
    'n_estimators':500,
    #'max_features': 0.5,
    'max_depth': 8,
    'min_samples_leaf': 2,
    'verbose': 0
}
# AdaBoost parameters
ada_params = {
    'n_estimators': 500,
    'learning_rate' : 0.75
}
# Gradient Boosting parameters
gb_params = {
    'n_estimators': 500,
     #'max_features': 0.2,
    'max_depth': 5,
    'min_samples_leaf': 2,
    'verbose': 0
}
# Support Vector Classifier parameters
svc_params = {
    'kernel' : 'linear',
    'C' : 0.025
    }

# Create 5 objects that represent our 4 models
# assign the class SklearnHelper to each ones
rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)

# Create Numpy arrays of train, test and target ( Survived)
# dataframes to feed into our models
y_train = train['Survived'].ravel() # ravel is same to reshape
train = train.drop(['Survived'], axis=1)
x_train = train.values # Creates an array of the train data
x_test = test.values # Creats an array of the test data

#%%
    # Output of the First level Predictions

# We now feed the training and test data into our 5 base classifiers and
# use the Out-of-Fold prediction function we defined earlier to generate
# our first level predictions

# Create our OOF train and test predictions. These base results will be used as new features
et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test) # Extra Trees
rf_oof_train, rf_oof_test = get_oof(rf,x_train, y_train, x_test) # Random Forest
ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test) # AdaBoost
gb_oof_train, gb_oof_test = get_oof(gb,x_train, y_train, x_test) # Gradient Boost
svc_oof_train, svc_oof_test = get_oof(svc,x_train, y_train, x_test) # Support Vector Classifier

print("Training is complete")

#%%
# Feature importances generated from the different classifiers
#rf_features = rf.feature_importances(x_train,y_train)
#et_features = et.feature_importances(x_train, y_train)
#ada_features = ada.feature_importances(x_train, y_train)
#gb_features = gb.feature_importances(x_train,y_train)

rf_features = [0.10474135,  0.21837029,  0.04432652,  0.02249159,  0.05432591,  0.02854371
  ,0.07570305,  0.01088129 , 0.24247496,  0.13685733 , 0.06128402]
et_features = [ 0.12165657,  0.37098307  ,0.03129623 , 0.01591611 , 0.05525811 , 0.028157
  ,0.04589793 , 0.02030357 , 0.17289562 , 0.04853517,  0.08910063]
ada_features = [0.028 ,   0.008  ,      0.012   ,     0.05866667,   0.032 ,       0.008
  ,0.04666667 ,  0.     ,      0.05733333,   0.73866667,   0.01066667]
gb_features = [ 0.06796144 , 0.03889349 , 0.07237845 , 0.02628645 , 0.11194395,  0.04778854
  ,0.05965792 , 0.02774745,  0.07462718,  0.4593142 ,  0.01340093]
#%%
# Create a dataframe from the lists containing the feature importance data
# for easy plotting via the Plotly package.

cols = train.columns.values
# Create a dataframe with features
feature_dataframe = pd.DataFrame( {'features': cols,
     'Random Forest feature importances': rf_features,
     'Extra Trees  feature importances': et_features,
      'AdaBoost feature importances': ada_features,
    'Gradient Boost feature importances': gb_features
    })

# Interactive feature importances via Plotly scatterplots
# Scatter plot
trace = go.Scatter(
    y = feature_dataframe['Random Forest feature importances'].values,
    x = feature_dataframe['features'].values,
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 25,
#       size= feature_dataframe['AdaBoost feature importances'].values,
        #color = np.random.randn(500), #set color equal to a variable
        color = feature_dataframe['Random Forest feature importances'].values,
        colorscale='Portland',
        showscale=True
    ),
    text = feature_dataframe['features'].values
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'Random Forest Feature Importance',
    hovermode= 'closest',
#     xaxis= dict(
#         title= 'Pop',
#         ticklen= 5,
#         zeroline= False,
#         gridwidth= 2,
#     ),
    yaxis=dict(
        title= 'Feature Importance',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatter2010')

# Scatter plot
trace = go.Scatter(
    y = feature_dataframe['Extra Trees  feature importances'].values,
    x = feature_dataframe['features'].values,
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 25,
#       size= feature_dataframe['AdaBoost feature importances'].values,
        #color = np.random.randn(500), #set color equal to a variable
        color = feature_dataframe['Extra Trees  feature importances'].values,
        colorscale='Portland',
        showscale=True
    ),
    text = feature_dataframe['features'].values
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'Extra Trees Feature Importance',
    hovermode= 'closest',
#     xaxis= dict(
#         title= 'Pop',
#         ticklen= 5,
#         zeroline= False,
#         gridwidth= 2,
#     ),
    yaxis=dict(
        title= 'Feature Importance',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatter2010')

# Scatter plot
trace = go.Scatter(
    y = feature_dataframe['AdaBoost feature importances'].values,
    x = feature_dataframe['features'].values,
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 25,
#       size= feature_dataframe['AdaBoost feature importances'].values,
        #color = np.random.randn(500), #set color equal to a variable
        color = feature_dataframe['AdaBoost feature importances'].values,
        colorscale='Portland',
        showscale=True
    ),
    text = feature_dataframe['features'].values
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'AdaBoost Feature Importance',
    hovermode= 'closest',
#     xaxis= dict(
#         title= 'Pop',
#         ticklen= 5,
#         zeroline= False,
#         gridwidth= 2,
#     ),
    yaxis=dict(
        title= 'Feature Importance',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatter2010')

# Scatter plot
trace = go.Scatter(
    y = feature_dataframe['Gradient Boost feature importances'].values,
    x = feature_dataframe['features'].values,
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 25,
#       size= feature_dataframe['AdaBoost feature importances'].values,
        #color = np.random.randn(500), #set color equal to a variable
        color = feature_dataframe['Gradient Boost feature importances'].values,
        colorscale='Portland',
        showscale=True
    ),
    text = feature_dataframe['features'].values
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'Gradient Boosting Feature Importance',
    hovermode= 'closest',
#     xaxis= dict(
#         title= 'Pop',
#         ticklen= 5,
#         zeroline= False,
#         gridwidth= 2,
#     ),
    yaxis=dict(
        title= 'Feature Importance',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatter2010')

# Create the new column containing the average of values

feature_dataframe['mean'] = feature_dataframe.mean(axis= 1) # axis = 1 computes the mean row-wise
feature_dataframe.head(3)

#%%
# Plotly Barplot of Average Feature Importances
y = feature_dataframe['mean'].values
x = feature_dataframe['features'].values
data = [go.Bar(
            x= x,
             y= y,
            width = 0.5,
            marker=dict(
               color = feature_dataframe['mean'].values,
            colorscale='Portland',
            showscale=True,
            reversescale = False
            ),
            opacity=0.6
        )]

layout= go.Layout(
    autosize= True,
    title= 'Barplots of Mean Feature Importance',
    hovermode= 'closest',
#     xaxis= dict(
#         title= 'Pop',
#         ticklen= 5,
#         zeroline= False,
#         gridwidth= 2,
#     ),
    yaxis=dict(
        title= 'Feature Importance',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='bar-direct-labels')

#%%
    # Second-Level Predictions from the First-level Output¶

# First-level output as new features
# ----------------------------------------------------- #
# Having now obtained our first-level predictions, one can think of it as
# essentially building a new set of features to be used as training data for
# the next classifier. As per the code below, we are therefore having as our
# new columns the first-level predictions from our earlier classifiers and
# we train the next classifier on this.
# ----------------------------------------------------- #
base_predictions_train = pd.DataFrame( {'RandomForest': rf_oof_train.ravel(),
     'ExtraTrees': et_oof_train.ravel(),
     'AdaBoost': ada_oof_train.ravel(),
      'GradientBoost': gb_oof_train.ravel()
    })
base_predictions_train.head()

# Correlation Heatmap of the Second Level Training set

data = [
    go.Heatmap(
        z= base_predictions_train.astype(float).corr().values ,
        x=base_predictions_train.columns.values,
        y= base_predictions_train.columns.values,
          colorscale='Viridis',
            showscale=True,
            reversescale = True
    )
]
py.iplot(data, filename='labelled-heatmap')

# There have been quite a few articles and Kaggle competition winner
# stories about the merits of having trained models that are more
# uncorrelated with one another producing better scores.
x_train = np.concatenate(( et_oof_train, rf_oof_train, ada_oof_train,
                          gb_oof_train, svc_oof_train), axis=1)
x_test = np.concatenate(( et_oof_test, rf_oof_test, ada_oof_test,
                         gb_oof_test, svc_oof_test), axis=1)


#%%
 # Second level learning model via XGBoost¶

# ---------------------------------------------------------- #
# Here we choose the eXtremely famous library for boosted tree
# learning model, XGBoost. It was built to optimize large-scale
# boosted tree algorithms
# ---------------------------------------------------------- #
gbm = xgb.XGBClassifier(
    #learning_rate = 0.02,
    n_estimators= 2000,
    max_depth= 4,
    min_child_weight= 2,
    #gamma=1,
    gamma=0.9,
    subsample=0.8,
    colsample_bytree=0.8,
    objective= 'binary:logistic',
    nthread= -1,
    scale_pos_weight=1).fit(x_train, y_train)
predictions = gbm.predict(x_test)

#%%
# Producing the Submission file

# Generate Submission File
StackingSubmission = pd.DataFrame({ 'PassengerId': PassengerId,
                            'Survived': predictions })
StackingSubmission.to_csv("StackingSubmission.csv", index=False)

#%%
#Some additional steps that may be taken to improve one's score could be:
#
#Implementing a good cross-validation strategy in training the models to find optimal parameter values
#Introduce a greater variety of base models for learning. The more uncorrelated the results, the better the final score.
