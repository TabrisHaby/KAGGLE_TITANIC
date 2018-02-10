# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#%%

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings('ignore')

#%%
data = pd.read_csv(r'E:\Data_and_Script\Python_Script\titanic\train.csv')
# test = pd.read_csv(r'E:\Data_and_Script\Python_Script\titanic\test.csv')
x = data.drop('Survived')
y = data['Survived']
data.isnull().sum()

#%%
f,ax=plt.subplots(1,2,figsize=(18,8))
data['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Survived')
ax[0].set_ylabel('')
sns.countplot('Survived',data=data,ax=ax[1])
ax[1].set_title('Survived')
plt.show()

#%%
# plot sex vs survived
_,ax = plt.subplots(1,2,figsize = (18,8))
sns.barplot(x = 'Sex',y = 'Survived',data = data,ax = ax[0])
sns.countplot('Sex',hue = 'Survived',data = data,ax = ax[1])

#%%
# plot pclass vs survived
_,ax = plt.subplots(1,3,figsize = (16,8))
sns.countplot('Pclass',data = data,ax = ax[0])
sns.countplot('Pclass',hue = 'Survived',data = data,ax= ax[1])
sns.barplot(x = 'Pclass',y = 'Survived',data = data,ax = ax[2])

#%%
# sex vs pclass
sns.pointplot(x = 'Pclass',y = 'Survived',hue = 'Sex',data = data)

#%%
# plot Age vs survived
_,ax = plt.subplots(1,2,figsize = (10,5))
sns.violinplot(x = 'Pclass',y = 'Age',hue = 'Survived',data = data,ax = ax[0],
               split = True)
ax[0].set_title('Pclass vs Age')
sns.violinplot(x = 'Sex',y = 'Age',hue = 'Survived',data = data,ax = ax[1],
               split = True)
ax[1].set_title('Sex vs Age')

#%%
# Extract Title from name
Title = list()
for i in range(0,len(data)) :
    Title.append(data['Name'].iloc[i].split(',')[1].split('.')[0].strip())
data['Title'] = Title

#%%  
# replace Title
data['Title'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','the Countess',
    'Jonkheer','Col','Rev','Capt','Sir','Don'],
    ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other',
     'Mr','Mr','Mr'],inplace=True)
#%%
# Title vs Age
data.groupby('Title').Age.mean()

# Filling NaN Ages¶
## Assigning the NaN Values with the Ceil values of the mean ages
data.loc[(data.Age.isnull())&(data.Title=='Mr'),'Age']=33
data.loc[(data.Age.isnull())&(data.Title=='Mrs'),'Age']=36
data.loc[(data.Age.isnull())&(data.Title=='Master'),'Age']=5
data.loc[(data.Age.isnull())&(data.Title=='Miss'),'Age']=22
data.loc[(data.Age.isnull())&(data.Title=='Other'),'Age']=46

#%%
# plot Age vs Survived
g = sns.FacetGrid(data = data ,hue = 'Survived')
g.map(sns.distplot,'Age',bins = 30)
g.add_legend()
#%%
# Title vs Pclass vs Survived
sns.factorplot(x = 'Pclass',y = 'Survived',col = 'Title',data = data)

#%%
# Embarked
_,ax = plt.subplots(2,2)
sns.countplot(data['Embarked'],ax = ax[0,0])
sns.countplot(data['Embarked'],hue = data['Sex'],ax = ax[0,1])
sns.countplot(data['Embarked'],hue = data['Survived'],ax = ax[1,0])
sns.countplot(data['Embarked'],hue = data['Pclass'],ax = ax[1,1])

sns.factorplot('Pclass','Survived',hue = 'Sex',col = 'Embarked',data = data)

# fill NA in embarked
data['Embarked'].fillna('S',inplace = True)
data.Embarked.isnull().any()# Finally No NaN values

#%%
# SibSp and Parch
sns.barplot(x = 'SibSp', y = 'Survived',data = data)
sns.barplot(x = 'Parch', y = 'Survived',data = data)

#%%
# Fare
# sns.factorplot(x = 'Fare',col = 'Pclass',data = data,kind = 'count')
g = sns.FacetGrid(data = data,col = 'Pclass',aspect = .8,size = 5)
g.map(sns.distplot,'Fare')

#%%
# heatmap correlation map
sns.heatmap(data.corr(),annot = True)


#%%
# Part2: Feature Engineering and Data Cleaning

# Age_Band
data['Age_band']=0
data.loc[data['Age']<=16,'Age_band']=0
data.loc[(data['Age']>16)&(data['Age']<=32),'Age_band']=1
data.loc[(data['Age']>32)&(data['Age']<=48),'Age_band']=2
data.loc[(data['Age']>48)&(data['Age']<=64),'Age_band']=3
data.loc[data['Age']>64,'Age_band']=4
data.head(2)

# Ageband vs Survived vs Pclass
sns.factorplot('Age_band','Survived',data =data,kind = 'bar',col = 'Pclass')

#%%
# FamilySIze and alone
data['FSize'] = data['SibSp'] + data['Parch']
data['IsAlone'] = data['FSize'].map(lambda x : 1 if x == 0 else 0)

# plot
_,ax = plt.subplots(1,2)
sns.barplot(x = 'FSize', y = 'Survived',data = data,ax = ax[0])
sns.barplot(x = 'IsAlone', y ='Survived',data = data, ax = ax[1])

# plot
sns.factorplot('IsAlone','Survived',col = 'Pclass',hue = 'Sex',data = data)

#%%
# Fare_range
data['Fare_cat']=0
data.loc[data['Fare']<=7.91,'Fare_cat']=0
data.loc[(data['Fare']>7.91)&(data['Fare']<=14.454),'Fare_cat']=1
data.loc[(data['Fare']>14.454)&(data['Fare']<=31),'Fare_cat']=2
data.loc[(data['Fare']>31)&(data['Fare']<=513),'Fare_cat']=3

# plot
sns.factorplot('Fare_cat','Survived',data=data,hue='Sex')
plt.show()

#%%
# convert to numeric value
data['Sex'].replace(['male','female'],[0,1],inplace=True)
data['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)
data['Title'].replace(['Mr','Mrs','Miss','Master','Other'],
    [0,1,2,3,4],inplace=True)

#%%
# drop unneed ed features
# PassengerId,Name,Age,Ticket,Fare,Cabin
data.drop(['Name','Age','Ticket','Fare','Cabin','PassengerId'],
          axis=1,inplace=True)

sns.heatmap(data.corr(),annot = True)


#%%
# Part3: Predictive Modeling¶

# ------------------------------------ #
1)Logistic Regression

2)Support Vector Machines(Linear and radial)

3)Random Forest

4)K-Nearest Neighbours

5)Naive Bayes

6)Decision Tree

7)Logistic Regression
# ------------------------------------ #

#%%
#importing all the required ML packages
from sklearn.linear_model import LogisticRegression #logistic regression
from sklearn import svm #support vector Machine
from sklearn.ensemble import RandomForestClassifier #Random Forest
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn.naive_bayes import GaussianNB #Naive bayes
from sklearn.tree import DecisionTreeClassifier #Decision Tree
from sklearn.model_selection import train_test_split #training and testing data split
from sklearn import metrics #accuracy measure
from sklearn.metrics import confusion_matrix #for confusion matrix

# train test split
x_train,x_test,y_train,y_test = train_test_split(data.drop(['Survived'],axis = 1),
                                                 data['Survived'],
                                                 test_size = .3,
                                                 random_state = 11)

#%%
# Radial Support Vector Machines(rbf-SVM)¶
svc = svm.SVC(gamma = .1,C = 1)
svc.fit(x_train,y_train)
y_pred1 = svc.predict(x_test)
print('The Accurary of SVC model is: ', metrics.accuracy_score(y_pred1,y_test))

# Linear Support Vector Machine(linear-SVM)¶
svc = svm.SVC(gamma = .1,C = 1,kernel = 'linear')
svc.fit(x_train,y_train)
y_pred2 = svc.predict(x_test)
print('The Accurary of Linear SVC model is: ', 
      metrics.accuracy_score(y_pred2,y_test))

# Logistic Regression¶
lreg = LogisticRegression()
lreg.fit(x_train,y_train)
y_pred3 = lreg.predict(x_test)
print('The Accurary of Logistic Regression model is: ', 
      metrics.accuracy_score(y_pred3,y_test))

# Decision Tree
dtc = DecisionTreeClassifier()
dtc.fit(x_train,y_train)
y_pred4 = dtc.predict(x_test)
print('The Accurary of Decision Tree model is: ', 
      metrics.accuracy_score(y_pred4,y_test))

# K-Nearest Neighbours(KNN)¶
knn = KNeighborsClassifier(n_neighbors = 9)
knn.fit(x_train,y_train)
y_pred5 = knn.predict(x_test)
print('The Accurary of KNN model is: ', metrics.accuracy_score(y_pred5,y_test),
      'Where n_neighbors = 9')

# Gaussian Naive Bayes¶
naiveB = GaussianNB()
naiveB.fit(x_train,y_train)
y_pred6 = naiveB.predict(x_test)
print('The Accurary of Gaussian Naive Bayes model is: ', 
      metrics.accuracy_score(y_pred6,y_test))

# Random Forests
rf = RandomForestClassifier()
rf.fit(x_train,y_train)
y_pred7 = rf.predict(x_test)
print('The Accurary of Random Forest model is: ', 
      metrics.accuracy_score(y_pred7,y_test))


#%%
# check n_neighbours value for knn model
a = list()
for i in list(range(1,11)) :
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(x_train,y_train)
    pred = knn.predict(x_test)
    a.append(metrics.accuracy_score(pred,y_test))

sns.barplot(list(range(1,11)),a,v)
# max value is where n_neighbors = 9

#%%

    # Cross Validation

from sklearn.model_selection import KFold #for K-fold cross validation
from sklearn.model_selection import cross_val_score #score evaluation
from sklearn.model_selection import cross_val_predict #prediction
kfold = KFold(n_splits=10, random_state=22) # k=10, split the data into 10 equal parts

classifiers=['Linear Svm','Radial Svm','Logistic Regression','KNN',
             'Decision Tree','Naive Bayes','Random Forest']
models=[svm.SVC(kernel='linear'),svm.SVC(kernel='rbf'),LogisticRegression(),
        KNeighborsClassifier(n_neighbors=9),DecisionTreeClassifier(),
        GaussianNB(),RandomForestClassifier(n_estimators=100)]
scores_cv = list()
score_mean = list()
score_std = list()
for i in models :
    model = i
    cv_result = cross_val_score(model,data.drop(['Survived'],axis = 1),y,
                                                cv = kfold,scoring = 'accuracy')
    scores_cv.append(cv_result)
    score_mean.append(cv_result.mean())
    score_std.append(cv_result.std())

model_cv_score = pd.DataFrame({'CV MEAN': score_mean, 'CV STD': score_std},
                              index = classifiers )
print(model_cv_score)

_,ax = plt.subplots(1,2)
sns.boxplot(x = classifiers, y = scores_cv,ax = ax[0])
sns.barplot(y = classifiers,x = score_mean,ax = ax[1])

# best two : Radial SVM and RandomForests.

#%%
# Hyper-Parameters Tuning

# SVM
from sklearn.model_selection import GridSearchCV
C=[0.05,0.1,0.2,0.3,0.25,0.4,0.5,0.6,0.7,0.8,0.9,1]
gamma=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
kernel=['rbf','linear']
hyper={'kernel':kernel,'C':C,'gamma':gamma}

gd = GridSearchCV(estimator = svm.SVC(),param_grid = hyper,verbose = True)
gd.fit(data.drop(['Survived'],axis = 1),y)

print('Best Score:', gd.best_score_)
print('Best Estimator:',gd.best_estimator_)

#Best Score: 0.828282828283
#Best Estimator: SVC(C=0.5, cache_size=200, class_weight=None, coef0=0.0,
#  decision_function_shape='ovr', degree=3, gamma=0.1, kernel='rbf',
#  max_iter=-1, probability=False, random_state=None, shrinking=True,
#  tol=0.001, verbose=False)

# Random Forest

n_estimators=range(100,1000,100)
hyper={'n_estimators':n_estimators}
gd=GridSearchCV(estimator=RandomForestClassifier(random_state=0),param_grid=hyper,verbose=True)
gd.fit(data.drop(['Survived'],axis = 1),y)
print(gd.best_score_)
print(gd.best_estimator_)

#0.817059483726
#RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#            max_depth=None, max_features='auto', max_leaf_nodes=None,
#            min_impurity_decrease=0.0, min_impurity_split=None,
#            min_samples_leaf=1, min_samples_split=2,
#            min_weight_fraction_leaf=0.0, n_estimators=900, n_jobs=1,
#            oob_score=False, random_state=0, verbose=0, warm_start=False)

#%%
# Ensembling

#Ensembling can be done in ways like:
#
#1)Voting Classifier
#
#2)Bagging
#
#3)Boosting.

#%%
# Voting Classifier

#It is the simplest way of combining predictions from many different simple 
#machine learning models. It gives an average prediction result based on the 
#prediction of all the submodels. The submodels or the basemodels are all of 
#diiferent types.

from sklearn.ensemble import VotingClassifier
ensemble_lin_rbf=VotingClassifier(estimators=[('KNN',KNeighborsClassifier(n_neighbors=10)),
                                              ('RBF',svm.SVC(probability=True,kernel='rbf',C=0.5,gamma=0.1)),
                                              ('RFor',RandomForestClassifier(n_estimators=500,random_state=0)),
                                              ('LR',LogisticRegression(C=0.05)),
                                              ('DT',DecisionTreeClassifier(random_state=0)),
                                              ('NB',GaussianNB()),
                                              ('svm',svm.SVC(kernel='linear',probability=True))
                                             ], 
                       voting='soft').fit(x_train,y_train)
print('The accuracy for ensembled model is:',ensemble_lin_rbf.score(x_test,y_test))
cross=cross_val_score(ensemble_lin_rbf,data.drop(['Survived'],axis = 1),y,
                       cv = 10,scoring = "accuracy")
print('The cross validated score is',cross.mean())

#The accuracy for ensembled model is: 0.85447761194
#The cross validated score is 0.823766031097

#%%
# Bagging

#Bagging is a general ensemble method. It works by applying similar 
#classifiers on small partitions of the dataset and then taking the average 
#of all the predictions. Due to the averaging,there is reduction in variance. 
#Unlike Voting Classifier, Bagging makes use of similar classifiers.

# Bagging KNN
from sklearn.ensemble import BaggingClassifier
bag_knn_fit = BaggingClassifier(base_estimator = KNeighborsClassifier(n_neighbors = 3),
                          random_state = 1,n_estimators = 700).fit(x_train,y_train)
y_pred = bag_knn_fit.predict(x_test)
print('The Accuracy of bagging KNN is :', metrics.accuracy_score(y_pred,y_test))
result = cross_val_score(ensemble_lin_rbf,data.drop(['Survived'],axis = 1),y,
                       cv = 10,scoring = "accuracy")
print('The mean accuracy of KNN CV is :',result.mean())

#The Accuracy of bagging KNn is : 0.832089552239
#The mean accuracy of CV is : 0.823766031097
#%%
# Bagging Decsion Tree
model=BaggingClassifier(base_estimator=DecisionTreeClassifier(),random_state=0,n_estimators=100)
model.fit(x_train,y_train)
prediction=model.predict(x_test)
print('The accuracy for bagged Decision Tree is:',metrics.accuracy_score(prediction,y_test))
result = cross_val_score(ensemble_lin_rbf,data.drop(['Survived'],axis = 1),y,
                       cv = 10,scoring = "accuracy")
print('The cross validated score for bagged Decision Tree is:',result.mean())

#The accuracy for bagged Decision Tree is: 0.832089552239
#The cross validated score for bagged Decision Tree is: 0.823766031097

#%%
# Boosting

#Boosting is an ensembling technique which uses sequential learning 
#of classifiers. It is a step by step enhancement of a weak model.

# AdaBoost(Adaptive Boosting)¶

from sklearn.ensemble import AdaBoostClassifier
ada=AdaBoostClassifier(n_estimators=200,random_state=0,learning_rate=0.1)
result=cross_val_score(ada,data.drop(['Survived'],axis = 1),y,
                       cv=10,scoring='accuracy')
print('The cross validated score for AdaBoost is:',result.mean())

#The cross validated score for AdaBoost is: 0.824952616048

# Stochastic Gradient Boosting

from sklearn.ensemble import GradientBoostingClassifier
grad=GradientBoostingClassifier(n_estimators=500,random_state=0,learning_rate=0.1)
result=cross_val_score(grad,data.drop(['Survived'],axis = 1),y
                       ,cv=10,scoring='accuracy')
print('The cross validated score for Gradient Boosting is:',result.mean())

# The cross validated score for Gradient Boosting is: 0.818286233118

# XGBoost

import xgboost as xg
xgboost=xg.XGBClassifier(n_estimators=900,learning_rate=0.1)
result=cross_val_score(xgboost,data.drop(['Survived'],axis = 1),y
                       ,cv=10,scoring='accuracy')
print('The cross validated score for XGBoost is:',result.mean())

# The cross validated score for XGBoost is: 0.810471002156

#%%

# We got the highest accuracy for AdaBoost. We will try to increase it 
# with Hyper-Parameter Tuning

# Hyper-Parameter Tuning for AdaBoost
n_estimators=list(range(100,1100,100))
learn_rate=[0.05,0.1,0.2,0.3,0.25,0.4,0.5,0.6,0.7,0.8,0.9,1]
hyper={'n_estimators':n_estimators,'learning_rate':learn_rate}
gd=GridSearchCV(estimator=AdaBoostClassifier(),param_grid=hyper,verbose=True)
gd.fit(data.drop(['Survived'],axis = 1),y)
print(gd.best_score_)
print(gd.best_estimator_)

#0.83164983165
#AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
#          learning_rate=0.05, n_estimators=200, random_state=None)

#%%
# Feature Importance¶

f,ax=plt.subplots(2,2,figsize=(15,12))
model=RandomForestClassifier(n_estimators=500,random_state=0)
model.fit(data.drop(['Survived'],axis = 1),y)
pd.Series(model.feature_importances_,data.drop(['Survived'].columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax[0,0])
ax[0,0].set_title('Feature Importance in Random Forests')
model=AdaBoostClassifier(n_estimators=200,learning_rate=0.05,random_state=0)
model.fit(data.drop(['Survived'],axis = 1),y)
pd.Series(model.feature_importances_,data.drop(['Survived'].columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax[0,1],color='#ddff11')
ax[0,1].set_title('Feature Importance in AdaBoost')
model=GradientBoostingClassifier(n_estimators=500,learning_rate=0.1,random_state=0)
model.fit(data.drop(['Survived'],axis = 1),y)
pd.Series(model.feature_importances_,data.drop(['Survived'].columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax[1,0],cmap='RdYlGn_r')
ax[1,0].set_title('Feature Importance in Gradient Boosting')
model=xg.XGBClassifier(n_estimators=900,learning_rate=0.1)
model.fit(data.drop(['Survived'],axis = 1),y)
pd.Series(model.feature_importances_,data.drop(['Survived'].columns).sort_values(ascending=True).plot.barh(width=0.8,ax=ax[1,1],color='#FD0F00')
ax[1,1].set_title('Feature Importance in XgBoost')
plt.show()















    

















