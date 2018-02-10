# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 14:25:46 2018

This is the second code
link : https://www.kaggle.com/startupsci/titanic-data-science-solutions
Titanic Data
Version 2


@author: Haby
"""

#%%
# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

# import train and test
train_df = pd.read_csv(r'E:\Data_and_Script\Python_Script\titanic\train.csv')
test_df = pd.read_csv(r'E:\Data_and_Script\Python_Script\titanic\test.csv')
combine = [train_df, test_df]

# data type for each data
train_df.info()
print('_'*40)
test_df.info()
print('_'*40)
train_df.describe() 
print('_'*40)
train_df.describe(include = ['O'])

#%%
train_df[['Pclass', 'Survived']].groupby(['Pclass'], 
        as_index=True).mean().sort_values(by='Survived', ascending=False)
#%%
train_df[["Sex", "Survived"]].groupby(['Sex'], 
        as_index=False).mean().sort_values(by='Survived', ascending=False)
#%%
train_df[["SibSp", "Survived"]].groupby(['SibSp'], 
        as_index=False).mean().sort_values(by='Survived', ascending=False)
#%%
train_df[["Parch", "Survived"]].groupby(['Parch'], 
        as_index=False).mean().sort_values(by='Survived', ascending=False)

#%%
# Visualization
sns.FacetGrid(train_df, col='Survived').map(plt.hist,'Age',bins = 20)
#%%
# grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();
#%%
# grid = sns.FacetGrid(train_df, col='Embarked')
grid = sns.FacetGrid(train_df,row = 'Embarked',size = 2.2,aspect = 1.6)
grid.map(sns.pointplot,'Pclass','Survived','Sex',palette = 'deep')
grid.add_legend()
#%%
# grid = sns.FacetGrid(train_df, col='Embarked', hue='Survived', palette={0: 'k', 1: 'w'})
grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()

#%%
# Droppinf some features
print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)

train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]

"After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape

#%%
# Create new features from exsiting

# Title
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train_df['Title'], train_df['Sex'])

#%%
# Change some title
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

#%%
# Convert some title to numeric
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train_df.head()

#%%
# Drop name safely
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]
train_df.shape, test_df.shape

#%%
# converting gender to numeric 
for dataset in combine :
    dataset['Sex'] = dataset['Sex'].map({'male' : 1,'female':0}).astype(int)
train_df.head()

#%%
# grid = sns.FacetGrid(train_df, col='Pclass', hue='Gender')
grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()


#%%

# set up a new 2 by 3 matrix to get the median age of differnet 
# Sex/Pclass group
guess_ages = np.zeros((2,3))

for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & \
                                  (dataset['Pclass'] == j+1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5

# fill the Na of age with the location of array in guess_ages, the location
# is made up by the sex(0,1) and pclass(1,2,3) and the position tuple(i,j)
# is made by the sex and pclass            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\
                    'Age'] = guess_ages[i,j]
    dataset['Age'] = dataset['Age'].astype(int)

train_df.head()

#%%
# Create ageband 
train_df['AgeBand'] = pd.cut(train_df['Age'],5)
train_df[['AgeBand','Survived']].groupby('AgeBand').mean()

#%%
# Replace Age with Age band
for dataset in combine :
    dataset.loc[dataset['Age'] <= 16,'Age'] = 0
    dataset.loc[(16 < dataset['Age']) & (dataset['Age'] <= 32),'Age'] = 1
    dataset.loc[(32 < dataset['Age']) & (dataset['Age'] <= 48),'Age'] = 2
    dataset.loc[(48 < dataset['Age']) & (dataset['Age'] <= 64),'Age'] = 3
    dataset.loc[(dataset['Age'] > 64),'Age'] = 4    

# drop AgeBand
train_df = train_df.drop(['AgeBand'],axis = 1)
combine = [train_df,test_df]
train_df.head()

#%%
# Create new feature based on existing features
for dataset in combine :
    dataset['FamilySize'] = dataset['Parch'] + dataset['SibSp'] + 1

train_df[['FamilySize','Survived']].groupby('FamilySize').mean()

#%%
# Create new variable ISALONE
for dataset in combine :
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1,'IsAlone'] = 1
    
train_df[['IsAlone','Survived']].groupby('IsAlone').mean()

#%%
# drop parch sibsp familysize since we have isalone
train_df = train_df.drop(['Parch','SibSp','FamilySize'],axis = 1)
test_df = test_df.drop(['Parch','SibSp','FamilySize'],axis = 1)

combine = [train_df,test_df]
train_df.head()

#%%
# New Feature Age * Pclass
for dataset in combine :
    dataset['AP'] = dataset['Age'] *dataset['Pclass']
train_df.loc[:,['AP','Pclass','Age']].head()

#%%
# Deal with categorical variables
for dataset in combine :
    dataset['Embarked'] = dataset['Embarked'].fillna('S')

train_df[['Embarked','Survived']].groupby('Embarked').mean()

#%%
# Convert to numeric variables
for dataset in combine :
    dataset['Embarked'] = dataset['Embarked'].map({'S':0,'C':1,'Q':2}).astype(int)
train_df.head()

#%%
# Fill NA of Fare with medians
test_df['Fare'].fillna(test_df['Fare'].dropna().median(),inplace = True)
test_df.head()

#%%
# FareBand based on Fare
train_df['FareBand'] = pd.qcut(train_df['Fare'],4)
train_df[['FareBand','Survived']].groupby('FareBand').mean().sort_values('Survived')

#%%
# Convert to numeric
for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]
    
train_df.head(10)


#%%
# Model Predict and Solve
# -------------------------------------------- #
#   Logistic Regression
#   KNN or k-Nearest Neighbors
#   Support Vector Machines
#   Naive Bayes classifier
#   Decision Tree
#   Random Forrest
#   Perceptron
#   Artificial neural network
#   RVM or Relevance Vector Machine
# -------------------------------------------- #


X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape

#%%
# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train,Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train,Y_train)*100,2)
print(acc_log)

coef_df = pd.DataFrame(train_df.columns.delete([0]))
coef_df.columns = ['Feature']
coef_df['Correlations'] = pd.Series(logreg.coef_[0])
coef_df.sort_values('Correlations',ascending = False)

#%%
# SVC
svc = SVC()
svc.fit(X_train,Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train,Y_train)*100,2)
print(accsvc)

#%%
# KNN
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn

#%%
# Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian

#%%
# Perceptron

perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
acc_perceptron

#%%
# Linear SVC

linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
acc_linear_svc

#%%
# Stochastic Gradient Descent

sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
acc_sgd

#%%
# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree

#%%
# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest

#%%

models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_linear_svc, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)

#%%

submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
# submission.to_csv('../output/submission.csv', index=False)
    