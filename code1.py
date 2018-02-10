# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 10:42:15 2018


This is the first Code written by python
About Titanic data
Version 1


@author: Haby
""" 
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore') 

#%%
# import train and test
train = pd.read_csv(r'E:\Data_and_Script\Python_Script\titanic\train.csv')
test = pd.read_csv(r'E:\Data_and_Script\Python_Script\titanic\test.csv')

# describe train
train.describe(include = 'all')

#%%
print(train.columns)
print(train.head())
print(train.shape)

# check NAs
print(train.isnull().sum())

#%%
# Basic EDA
# Sex
# draw picture of sex vs survived rate
sns.barplot(x = 'Sex',y = 'Survived',data = train)
#%%
# Pclass Feature
# draw barplot of survived by pclass
sns.barplot(x = 'Pclass',y  ='Survived',data = train)
#%%
# SibSp
#draw a bar plot for SibSp vs. survival
sns.barplot(x="SibSp", y="Survived", data=train)
#%%
# parch
#draw a bar plot for parch vs. survival
sns.barplot(x="Parch", y="Survived", data=train)
#%%
# Age feature
train['Age'] = train['Age'].fillna(-0.5)
test['Age'] = test['Age'].fillna(-0.5)
bins = [-1,0,5,12,18,24,35,60,np.inf]
label = ['Unknown','Baby','Child','Teenager','Student','Young Adult','Adult',
         'Senior']
train['AgeGroup'] = pd.cut(train['Age'],bins = bins,labels = label)
test['AgeGroup'] = pd.cut(test['Age'],bins = bins,labels = label)

# plot new group
sns.barplot(x = 'AgeGroup',y = 'Survived',data = train)
plt.show()
#%%
# Cabin Feature
train['CabinPool'] = (train['Cabin'].notnull().astype('int'))
test['CabinPool'] = (train['Cabin'].notnull().astype('int'))

# plot
sns.barplot(x = 'CabinPool', y = 'Survived', data = train)
plt.show()


#%%
# Cleaning Data

# Drop Cabin Feature
train = train.drop(['Cabin'],axis = 1)
test = test.drop(['Cabin'],axis = 1)

# Drop Ticket Featrue
train = train.drop(['Ticket'],axis = 1)
test = test.drop(['Ticket'],axis = 1)

# Embarked Features
print(train.groupby('Embarked')['Embarked'].value_counts())

# replace missings with S since majority of pple from S
train['Embarked'] = train['Embarked'].fillna('S')

#%%
# Age Features

# create a combined group for both data
combine = [train,test]

# extract a title for each Name in the train and test datasets
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.',
           expand = False)

pd.crosstab(train['Title'], train['Sex'])

# replace various titles with more common names
for dataset in combine :
    dataset['Title'] = dataset['Title'].replace(['Lady','Capt','Col',
           'Don','Dr','Major','Rev','Jonkheer','Dona'],'Rare')
    dataset['Title'] = dataset['Title'].replace(['Countess','Lady','Sir'],
           'Royal')
    dataset['Title'] = dataset['Title'].replace('Mlle','Miss')
    dataset['Title'] = dataset['Title'].replace('Ms','Miss')
    dataset['Title'] = dataset['Title'].replace('Mme','Mrs')

train[['Title','Survived']].groupby(['Title'],as_index = False).mean()

# map each title groups with a numeric value
title_mapping = {'Mr':1,'Miss':2,'Mrs':3,'Master':4,'Royal':5,'Rare':6}
for dataset in combine :
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
train.head()

#%%
# Perdict missing values
# fill missing age with mode age group for each title
mr_age = train[train["Title"] == 1]["AgeGroup"].mode() #Young Adult
miss_age = train[train["Title"] == 2]["AgeGroup"].mode() #Student
mrs_age = train[train["Title"] == 3]["AgeGroup"].mode() #Adult
master_age = train[train["Title"] == 4]["AgeGroup"].mode() #Baby
royal_age = train[train["Title"] == 5]["AgeGroup"].mode() #Adult
rare_age = train[train["Title"] == 6]["AgeGroup"].mode() #Adult

age_title_mapping = {1: "Young Adult", 2: "Student", 3: "Adult", 4: "Baby", 5: "Adult", 6: "Adult"}

for x in range(len(train["AgeGroup"])):
    if train["AgeGroup"][x] == "Unknown":
        train["AgeGroup"][x] = age_title_mapping[train["Title"][x]]
        
for x in range(len(test["AgeGroup"])):
    if test["AgeGroup"][x] == "Unknown":
        test["AgeGroup"][x] = age_title_mapping[test["Title"][x]]

#%%
#map each Age value to a numerical value
age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4, 'Young Adult': 5, 'Adult': 6, 'Senior': 7}
train['AgeGroup'] = train['AgeGroup'].map(age_mapping)
test['AgeGroup'] = test['AgeGroup'].map(age_mapping)

train.head()

#dropping the Age feature for now, might change
train = train.drop(['Age'], axis = 1)
test = test.drop(['Age'], axis = 1)

# dropping name feature
train = train.drop(['Name'],axis = 1)
test = test.drop(['Name'],axis = 1)


#%%
# Sex feature to numeric
sex_mapping = {'male' : 0,'female' : 1}
train['Sex'] = train['Sex'].map(sex_mapping)
test['Sex'] = test['Sex'].map(sex_mapping)

train.head()

#%%
# Embarked feature
embark_mapping = {'S':1,'C':2,'Q':3}
train['Embarked'] = train['Embarked'].map(embark_mapping)
test['Embarked'] = test['Embarked'].map(embark_mapping)

train.head()

#%%
#fill in missing Fare value in test set based on mean fare for that Pclass 
for x in range(len(test["Fare"])):
    if pd.isnull(test["Fare"][x]):
        pclass = test["Pclass"][x] #Pclass = 3
        test["Fare"][x] = round(train[train["Pclass"] == pclass]["Fare"].mean(), 4)
        
#map Fare values into groups of numerical values
train['FareBand'] = pd.qcut(train['Fare'], 4, labels = [1, 2, 3, 4])
test['FareBand'] = pd.qcut(test['Fare'], 4, labels = [1, 2, 3, 4])

#drop Fare values
train = train.drop(['Fare'], axis = 1)
test = test.drop(['Fare'], axis = 1)

train.head()   
test.head() 

#%%
# Choose the best model

# split the train data for test part
from sklearn.model_selection import train_test_split

predictors = train.drop(['Survived','PassengerId'],axis = 1)
target = train['Survived']
x_train,x_val,y_train,y_val = train_test_split(predictors,target,
                                               test_size = 0.22,random_state = 0)

#%%

#Testing Different Models

# -------------------------------------------------------------- #
#   I will be testing the following models with my training data:
#       Gaussian Naive Bayes
#       Logistic Regression
#       Support Vector Machines
#       Perceptron
#       Decision Tree Classifier
#       Random Forest Classifier
#       KNN or k-Nearest Neighbors
#       Stochastic Gradient Descent
#       Gradient Boosting Classifier
# -------------------------------------------------------------- #
#%%
# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
y_pred = gaussian.predict(x_val)
acc_gaussian = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_gaussian)

#%%
# Logistic Regression
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_val)
acc_logreg = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_logreg)

#%%
# Support Vector Machines
from sklearn.svm import SVC

svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_val)
acc_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_svc)

#%%
# Linear SVC
from sklearn.svm import LinearSVC

linear_svc = LinearSVC()
linear_svc.fit(x_train, y_train)
y_pred = linear_svc.predict(x_val)
acc_linear_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_linear_svc)

#%%
# Perceptron
from sklearn.linear_model import Perceptron

perceptron = Perceptron()
perceptron.fit(x_train, y_train)
y_pred = perceptron.predict(x_val)
acc_perceptron = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_perceptron)

#%%
#Decision Tree
from sklearn.tree import DecisionTreeClassifier

decisiontree = DecisionTreeClassifier()
decisiontree.fit(x_train, y_train)
y_pred = decisiontree.predict(x_val)
acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_decisiontree)

#%%
# Random Forest
from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier()
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_val)
acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_randomforest)

#%%
# KNN or k-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
y_pred = knn.predict(x_val)
acc_knn = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_knn)

#%%
# Stochastic Gradient Descent
from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier()
sgd.fit(x_train, y_train)
y_pred = sgd.predict(x_val)
acc_sgd = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_sgd)

#%%
# Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier

gbk = GradientBoostingClassifier()
gbk.fit(x_train, y_train)
y_pred = gbk.predict(x_val)
acc_gbk = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_gbk)

#%%
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 'Linear SVC', 
              'Decision Tree', 'Stochastic Gradient Descent', 'Gradient Boosting Classifier'],
    'Score': [acc_svc, acc_knn, acc_logreg, 
              acc_randomforest, acc_gaussian, acc_perceptron,acc_linear_svc, acc_decisiontree,
              acc_sgd, acc_gbk]})
models.sort_values(by='Score', ascending=False)

#%%
#set ids as PassengerId and predict survival 
ids = test['PassengerId']
predictions = rf_cv.predict(test.drop('PassengerId', axis=1))

#set the output as a dataframe and convert to csv file named submission.csv
output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
output.to_csv('submission.csv', index=False)