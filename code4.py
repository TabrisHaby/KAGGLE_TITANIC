# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 18:12:10 2018

Titanic Data
version 3

@author: Kaggle Kernel
"""
#%%
# import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
# import ml packages
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

#%%
# import data
train = pd.read_csv(r'E:\Data_and_Script\Python_Script\titanic\train.csv')
test = pd.read_csv(r'E:\Data_and_Script\Python_Script\titanic\test.csv')

# preview data
# train.head()

train.info()
print('-'*40)
test.info()
#%%
    # Embarked

# fill the 2 missing embarked value in train with S
train['Embarked'] = train['Embarked'].fillna('S')

# plot
sns.factorplot('Embarked','Survived',data = train)

#%%
fig,ax = plt.subplots(1,3,figsize = (15,5))

# counterplot
sns.countplot('Embarked',data = train,ax = ax[0])
sns.countplot('Survived', hue="Embarked", data=train, order=[1,0], ax=ax[1])

# group by embarked and get the mean value of Embarked
em_perc = train[['Embarked','Survived']].groupby(['Embarked'],
               as_index = False).mean()
sns.barplot(x = 'Embarked',y = 'Survived',data = em_perc,
              order = ['C','Q','S'],ax = ax[2])

#%%
# get dummy variables for embarked and drop embarked / S variables
embark_dummy_train = pd.get_dummies(train['Embarked'])
embark_dummy_train.drop(['S'],axis = 1,inplace = True)

embark_dummy_test = pd.get_dummies(test['Embarked'])
embark_dummy_test.drop(['S'],axis = 1,inplace = True)

train = train.join(embark_dummy_train)
test = test.join(embark_dummy_test)

train.drop(['Embarked'],axis = 1,inplace = True)
test.drop(['Embarked'],axis = 1,inplace = True)

#%%
    # Fare

# fill na for test data with median
test['Fare'].fillna(test.Fare.median(),inplace = True)

# convert float to int
train.Fare = train.Fare.astype(int)
test.Fare = test.Fare.astype(int)

f,ax = plt.subplots(1,2,figsize = (12,5))
# plot fare for survived/no survived ppl
sns.boxplot(x = np.log10(train.Survived),y = np.log(train.Fare),ax = ax[0])
# plot number of each fare
sns.distplot(train.Fare,bins = 50)

#%%
    # Age

fig,(ax1,ax2) = plt.subplots(1,2,figsize = (15,4))
ax1.set_title('Original Age Value')
ax2.set_title('New Age Value')

# get average, std, and number of NaN values in titanic_df
average_age_titanic   = train["Age"].mean()
std_age_titanic       = train["Age"].std()
count_nan_age_titanic = train["Age"].isnull().sum()

# get average, std, and number of NaN values in test_df
average_age_test   = test["Age"].mean()
std_age_test       = test["Age"].std()
count_nan_age_test = test["Age"].isnull().sum()

# generate random numbers between (mean - std) & (mean + std)
rand_1 = np.random.randint(average_age_titanic - std_age_titanic,
                           average_age_titanic + std_age_titanic,
                           size = count_nan_age_titanic)
rand_2 = np.random.randint(average_age_test - std_age_test,
                           average_age_test + std_age_test,
                           size = count_nan_age_test)

# plot original Age values
# NOTE: drop all null values, and convert to int
train['Age'].dropna().astype(int).hist(bins=70, ax=ax1)

# fill NaN values in Age column with random values generated
train["Age"][np.isnan(train["Age"])] = rand_1
test["Age"][np.isnan(test["Age"])] = rand_2

# convert from float to int
train['Age'] = train['Age'].astype(int)
test['Age']  = test['Age'].astype(int)

# plot new Age Values
train['Age'].hist(bins=70, ax=ax2)

#%%
# peaks for survived/not survived passengers by their age
facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, train['Age'].max()))
facet.add_legend()

# average survived passengers by age
fig, axis1 = plt.subplots(1,1,figsize=(18,4))
average_age = train[["Age", "Survived"]].groupby(['Age'],as_index=False).mean()
sns.barplot(x='Age', y='Survived', data=average_age)

#%%
    # Cabin
    # Delete Cabin since too many NAs
train.drop('Cabin',axis = 1,inplace = True)
test.drop('Cabin',axis = 1,inplace = True)

#%%
    # # Family

# Instead of having two columns Parch & SibSp,
# we can have only one column represent if the passenger had any
# family member aboard or not,
# Meaning, if having any family member(whether parent, brother, ...etc)
# will increase chances of Survival or not.

train['Family'] = train.Parch + train.SibSp
train[train['Family'] > 0]['Family'] = 1
train[train['Family'] == 0]['Family'] = 0

test['Family'] = test.Parch + test.SibSp
test[test['Family'] > 0]['Family'] = 1
test[test['Family'] == 0]['Family'] = 0

# drop SibSp/ Parch
train.drop(['SibSp','Parch'],axis = 1,inplace = True)
test.drop(['SibSp','Parch'],axis = 1,inplace = True)

# plot1
fig,(axis1,axis2) = plt.subplots(1,2,sharex = True,figsize = (10,5))

sns.countplot(x = train.Family,order = [1,0],ax = axis1)
# plot2
fam_perc = train[['Family','Survived']].groupby('Family',
                as_index = True).mean()
sns.barplot(x = 'Family',y = 'Survived',data = train,order = [1,0],
            ax = axis2)

axis1.set_xticklabels(['With Family','Alone'],rotation = 45)

#%%
    # Sex
    # As we see, children(age < ~16) on aboard seem to have a high
    # chances for Survival.
    # So, we can classify passengers as males, females, and child

def get_person(p) :
    age,sex = p
    return 'Child' if age < 16 else sex

train['Person'] = train[['Age','Sex']].apply(get_person,axis = 1)
test['Person'] = test[['Age','Sex']].apply(get_person,axis = 1)

# drop sex
train.drop('Sex',axis = 1,inplace = True)
test.drop('Sex',axis = 1,inplace = True)

# create dummy fro person
# drop Male as it has the lowest average of survived passengers
person_dummies_titanic  = pd.get_dummies(train['Person'])
person_dummies_titanic.columns = ['Child','Female','Male']
person_dummies_titanic.drop(['Male'], axis=1, inplace=True)
# ----------------------------------------------------- #
# 比如说现有有　小学　初中　高中　大学四个级别的学历，如果你设置了四个虚拟
# 变量，从变量的组成结构上，如果小学的设为1，其他为0，初中为1，其他为0，
# 高中为1其他为0，大学为1，其他为0。显然，四个变量和刚好等于[1]，回归时就
# 是完全共线性，stata软件自然要droped一个。这种情况叫“虚拟变量陷阱”，所以
# 你高3个虚拟变量就行了。
# -----------------------------------------------------  #
#%%
person_dummies_test  = pd.get_dummies(test['Person'])
person_dummies_test.columns = ['Child','Female','Male']
person_dummies_test.drop(['Male'], axis=1, inplace=True)

train = train.join(person_dummies_titanic)
test = test.join(person_dummies_test)

fig, (axis1,axis2) = plt.subplots(1,2,figsize=(10,5))
#%%
# sns.factorplot('Person',data=titanic_df,kind='count',ax=axis1)
sns.countplot(x='Person', data=train, ax=axis1)

# average of survived for each Person(male, female, or child)
person_perc = train[["Person", "Survived"]].groupby(['Person'],as_index=False).mean()
sns.barplot(x='Person', y='Survived', data=person_perc, ax=axis2,
            order=['male','female','child'])

train.drop(['Person'],axis=1,inplace=True)
test.drop(['Person'],axis=1,inplace=True)

#%%
    # Pclass

sns.factorplot('Pclass','Survived',order = [1,2,3],data = train,size = 5)

# get dummy
pclass_dummies_titanic  = pd.get_dummies(train['Pclass'])
pclass_dummies_titanic.columns = ['Class_1','Class_2','Class_3']
pclass_dummies_titanic.drop(['Class_3'], axis=1, inplace=True)

pclass_dummies_test  = pd.get_dummies(test['Pclass'])
pclass_dummies_test.columns = ['Class_1','Class_2','Class_3']
pclass_dummies_test.drop(['Class_3'], axis=1, inplace=True)

train.drop(['Pclass'],axis=1,inplace=True)
test.drop(['Pclass'],axis=1,inplace=True)

train = train.join(pclass_dummies_titanic)
test    = test.join(pclass_dummies_test)

#%%
# define training and testing sets

X_train = train.drop("Survived",axis=1)
Y_train = train["Survived"]
X_test  = test.drop("PassengerId",axis=1).copy()

#%%
    # Regression

    # LogisticRegression
reg = LogisticRegression()
reg.fit(X_train,Y_train)
Y_pred = reg.predict(X_test)
reg.score(X_train,Y_train)

    # Support Vector Machines

svc = SVC()

svc.fit(X_train, Y_train)

Y_pred = svc.predict(X_test)

svc.score(X_train, Y_train)

    # Random Forest
# Random Forests

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)
