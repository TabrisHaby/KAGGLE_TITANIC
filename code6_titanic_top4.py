# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 12:50:41 2018

Titanic Data
Version 6
https://www.kaggle.com/yassineghouzam/titanic-top-4-with-ensemble-modeling

@author: Haby
"""

        # Titanic Top 4% with ensemble modeling

#%%
        # ---------------------------------------------- #
        # -------------- Feature analysis -------------- #
        # ---------------------------------------------- #
        
#%%
# 1. import Packages and env
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from collections import Counter

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, \
        GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, \
        StratifiedKFold, learning_curve

sns.set(style='white', context='notebook', palette='deep')

#%%
# 2.1 Load data

train = pd.read_csv(r'E:\Data_and_Script\Python_Script\titanic\train.csv')
test = pd.read_csv(r'E:\Data_and_Script\Python_Script\titanic\test.csv')
IDtest = test.PassengerId

#%%
# 2.2 Outlier Detection
def detect_outliers(df,n,features) :
    """
    Takes a dataframe df of features and returns a list of the indices
    corresponding to the observations containing more than n outliers according
    to the Tukey method.
    """
    outlier_indices = []
    
    # iterate over features
    for c in features :
        # 25% / 75% quantile
        q1 = np.percentile(df[c],25)
        q3 = np.percentile(df[c],75)
        iqr = q3 - q1
        
        # outliers step
        outlier_step = 1.5*iqr
        # detemeter the list of index of outlier for each col
        outlier_list_col = df[(df[c] < q1 - outlier_step\
                               ) | (df[c] > q3 + outlier_step )].index
        # append the result to outlier_indices
        outlier_indices.extend(outlier_list_col)
    
    # select obs with more that 2 outliers
    outlier_indices = Counter(outlier_indices) # count the number of each item
    multiple_outliers = list(k for k,v in outlier_indices.items() if v > n)
    
    return multiple_outliers

outlier_to_drop = detect_outliers(train,2,['Age','SibSp','Parch','Fare'])

#%%
train.loc[outlier_to_drop]   
train.drop(outlier_to_drop,axis = 0,inplace = True  )    

#%%
# 2.3 joining train and test set
dataset =  pd.concat(objs=[train, test], axis=0).reset_index(drop=True)    
train_len = len(train)

#%%
# 2.4 Check for NAs 
dataset = dataset.fillna(np.nan)
dataset.isnull().sum()
 
# Infos
train.info()
train.describe()

#%%
# 3.1 Numerical values

# Correlation matrix between numerical values (SibSp Parch Age and Fare values) and Survived 
sns.heatmap(train[['Survived','SibSp','Parch','Age','Fare']].corr(),
            annot=True,fmt = '.3f') 
# Only Fare feature seems to have a significative correlation with the survival probability.   

#%%
# Explore other Numeric variables

# SibSp
f = sns.barplot(x = 'SibSp',y = 'Survived',data = train )    
f.set(ylabel = 'Survived Rate')    
# It seems that passengers having a lot of siblings/spouses have less chance 
# to survive.

# Single passengers (0 SibSP) or with two other persons (SibSP 1 or 2) have 
# more chance to survive

#%%
# Parch    
f = sns.barplot(x = 'Parch',y = 'Survived',data = train)
f.set(ylabel = 'Survived Rate')  
# Small families have more chance to survive, more than single (Parch 0), 
# medium (Parch 3,4) and large families (Parch 5,6 ).  

#%%
# Age vs Survived
g = sns.FacetGrid(col = 'Survived',data = train)
g.map(sns.distplot,'Age')  

#%%
# Explore Age distribution
temp = train[train.Age.notnull()][['Age','Survived']]
g = sns.kdeplot(temp[temp['Survived'] == 1]['Age'],shade = True,color = 'c')
g = sns.kdeplot(temp[temp['Survived'] == 0]['Age'],shade = True,color = 'r')    
g.legend(['Survived','Not Survived'])
# When we superimpose the two densities , we cleary see a peak correponsing 
# (between 0 and 5) to babies and very young childrens.    
    
#%%
# Fare

# imputation with median
dataset['Fare'] = dataset.Fare.fillna(dataset['Fare'].median()) 
# plot Fare  
sns.kdeplot(dataset['Fare'],color = 'c')  
    
# It is better to transform it with the log function to reduce this skew.
# Apply log to Fare to reduce skewness distribution
dataset["Fare"] = dataset["Fare"].map(lambda i: np.log(i) if i > 0 else 0)

#%%
# plot scaled fare
sns.kdeplot(dataset['Fare'],color = 'r')
# Skewness is clearly reduced after the log transformation

#%%
# 3.2 Categorical Variabls

# Sex
sns.barplot(x = 'Sex',y = 'Survived',data = train)

# survived rate for male / female
train[['Sex','Survived']].groupby('Sex').mean()
    
# Pclass
sns.barplot(x = 'Pclass',y = 'Survived',data = train,hue = 'Sex')  
train[['Pclass','Survived']].groupby('Pclass').mean()  
  
# Embarked
dataset['Embarked'] = dataset['Embarked'].fillna('S')   
#%% 
# plot
f,ax = plt.subplots(1,3,figsize = (10,5))
sns.barplot(x = 'Embarked', y = 'Survived',data = train,ax = ax[0])
sns.barplot(x = 'Embarked',y = 'Survived',data = train,hue = 'Pclass',ax = ax[1])        
sns.barplot(x = 'Embarked',y = 'Survived',data = train,hue = 'Sex',ax = ax[2    ])        

#%%
# 4 filling missing values  
      
# 4.1 Age   
_,ax = plt.subplots(1,4)
sns.boxplot(x = 'Sex',y = 'Age',data = train,ax = ax[0])     
sns.boxplot(x = 'Sex',y = 'Age',hue = 'Pclass',data = train,ax = ax[1]) 
sns.boxplot(x = 'Parch',y = 'Age',data = train,ax = ax[2])
sns.boxplot(x = 'SibSp',y = 'Age',data = train,ax = ax[3])       
        
# Age distribution seems to be the same in Male and Female subpopulations, 
# so Sex is not informative to predict Age.

# However, 1rst class passengers are older than 2nd class passengers who 
# are also older than 3rd class passengers.

# Moreover, the more a passenger has parents/children the older he is 
# and the more a passenger has siblings/spouses the younger he is.        

#%%
# convert sex tp 1 / 0 and plot
dataset['Sex'] = dataset['Sex'].map({'male' : 0,'female' : 1})
sns.heatmap(dataset[['Age','SibSp','Parch','Sex','Pclass']].corr(),
            annot = True)
# sex is not related with age
# use SibSP, Parch and Pclass in order to impute the missing ages.
# fill Age with the median age of similar rows of Pclass, Parch and SibSp.       

# findall nan index of Age
index_age_nan = list(dataset[dataset['Age'].isnull()].index)     

for i in index_age_nan :
    age_med = dataset['Age'].median()
    age_pred = dataset[((dataset['Pclass'] == dataset.iloc[i]['Pclass']) & \
                       (dataset['Parch'] == dataset.iloc[i]['Parch']) & \
                       (dataset['SibSp'] == dataset.iloc[i]['SibSp']))]['Age'].median()
    if not np.isnan(age_pred) :
        dataset['Age'].iloc[i] = age_pred
    else :
        dataset['Age'].iloc[i] = age_med
# ---------------------------------------------- #
# for i in index_age_nan :
#    age_med = dataset['Age'].median()
#    age_pred = dataset[((dataset['Pclass'] == dataset.iloc[i]['Pclass']) & \
#                       (dataset['Parch'] == dataset.iloc[i]['Parch']) & \
#                       (dataset['SibSp'] == dataset.iloc[i]['SibSp']))]['Age'].median()
#    if not np.isnan(age_pred) :
#        dataset.iloc[i]['Age'] = age_pred
#    else :
#        dataset.iloc[i]['Age'] = age_med  
---- > 不成立,先定位index后提供column时候，定位点在index，而不再column，
---- > 因此无法赋值

---- > list(dataset[dataset['Age'].isnull()].index) 输出为一个list值
---- > [dataset[dataset['Age'].isnull()].index] 输出值为包含序列的生成器
---- > Out[783]: Int64Index([], dtype='int64')
#%%
        # ---------------------------------------------- #
        # ------------ Feature engineering ------------- #
        # ---------------------------------------------- #
        
# 5.1 Name/ Title
dataset['Title'] = [i.split(',')[1].split('.')[0].strip() for i in dataset['Name']]

# plot
sns.countplot(dataset['Title'])       

# group into 4
dataset['Title'] = dataset['Title'].replace(['Lady', 'the Countess',
       'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 
       'Sir', 'Jonkheer', 'Dona'],'Rare')        
dataset["Title"] = dataset["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , 
       "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})
dataset["Title"] = dataset["Title"].astype(int)   

# plot again
sns.countplot(dataset['Title'])  
sns.barplot(x = 'Title',y = 'Survived',data = dataset)   
        
# Drop Name variable
dataset.drop(labels = ["Name"], axis = 1, inplace = True)        
        
#%%
# 5.2 Family Size
dataset['FamilySize'] = dataset['Parch'] + dataset['SibSp'] + 1

# plot
sns.barplot(x = 'FamilySize',y = 'Survived',data =dataset) 
sns.countplot(dataset['FamilySize'])

# group into 4 : single / couples / medfamily / largefamily
dataset['Single'] = dataset['FamilySize'].map(lambda x : 1 if x == 1 else 0) 
dataset['Couple'] = dataset['FamilySize'].map(lambda x : 1 if x == 2 else 0)      
dataset['MedF'] = dataset['FamilySize'].map(lambda x : 1 if 3<= x <= 4 else 0)      
dataset['LarF'] = dataset['FamilySize'].map(lambda x : 1 if x >= 5 else 0)  

# plot  
_,ax = plt.subplots(1,4)
sns.barplot(x = 'Single',y = 'Survived',data = dataset,ax = ax[0])
sns.barplot(x = 'Couple',y = 'Survived',data = dataset,ax = ax[1])
sns.barplot(x = 'MedF',y = 'Survived',data = dataset,ax = ax[2])
sns.barplot(x = 'LarF',y = 'Survived',data = dataset,ax = ax[3])
# Factorplots of family size categories show that Small and Medium families
# have more chance to survive than single passenger and large families.
    
#%%
# get dummy for title and embark
dataset = pd.get_dummies(dataset,columns = ['Title'])
dataset = pd.get_dummies(dataset,columns = ['Embarked'],prefix = 'Em')
        
#%%
# 5.3 Cabin

# check data
set(dataset.Cabin) 

# fill nan with X
dataset['Cabin'] = dataset['Cabin'].fillna('X')

# check some combined tickets
dataset.Cabin.apply(len).value_counts()
# 1     1013
# 3      184
# 4       32
# 2       32
# 7       19
# 11       7
# 5        7
# 15       5

# Capital Cabin except length with 5
dataset['Cabin'] = dataset['Cabin'].map(lambda x : x[0] if len(x) != 5 else x)

# for length = 5, they have Cabin with E,F and G 
# id = 129, i set cabin = E based on Fare and Sex
# id = 1180,1213 set cabin = F

# for no more evidence, just guess young adult in better cabin
dataset['Cabin'].loc[dataset['PassengerId'] == 1180] == 'F'
dataset['Cabin'].loc[dataset['PassengerId'] == 1213] == 'F'
dataset['Cabin'].loc[dataset['PassengerId'] == 129] == 'E'
dataset['Cabin'].loc[dataset['PassengerId'] == 76] == 'F'
dataset['Cabin'].loc[dataset['PassengerId'] == 700] == 'G'
dataset['Cabin'].loc[dataset['PassengerId'] == 716] == 'F'
dataset['Cabin'].loc[dataset['PassengerId'] == 938] == 'G'

# dummt
dataset = pd.get_dummies(dataset, columns = ["Cabin"],prefix="Cabin")


#%%
# 5.4 Tickets

# split title
t = []
for i in dataset.Ticket :
    if i.isdigit():
        t.append(i[:1])
    else :
        t.append(i.replace('.','').replace('/','').split()[0].upper())
# add to dataset   
dataset['Ticket'] = t

# group
for i in dataset.index :   
    if dataset['Ticket'].iloc[i] in ['STONO','STONO2','STONOQ','SOTONO2'] :
        dataset['Ticket'].iloc[i] = 'SOTONOQ'
    elif dataset['Ticket'].iloc[i] in ['SCAH','SC','SCA4','SCA3','SCOW'] :
        dataset['Ticket'].iloc[i] = 'SC'
    elif dataset['Ticket'].iloc[i] in ['FCC','6','SOC','C','SOPP',
              'PP','LINE','WEP','FC','5','8','PPP','9','SWPP','FA',
              'AQ3','LP','SOP','A','CASOTON','AS','SP','AQ4'] :
        dataset['Ticket'].iloc[i] = 'RARE'
# result
dataset.Ticket.value_counts()       
# dummy
dataset = pd.get_dummies(dataset, columns = ["Ticket"], prefix="T")

#%%
# get dummy
# Create categorical values for Pclass
dataset["Pclass"] = dataset["Pclass"].astype("category")
dataset = pd.get_dummies(dataset, columns = ["Pclass"],prefix="Pc")

# Drop useless variables 
dataset.drop(labels = ["PassengerId"], axis = 1, inplace = True)

#%%
dataset.shape

#%%               
        # ---------------------------------------------- #
        # ------------------ Modeling ------------------ #
        # ---------------------------------------------- #
        
## Separate train dataset and test dataset

train = dataset[:train_len]
test = dataset[train_len:]
test.drop(labels=["Survived"],axis = 1,inplace=True)

## Separate train features and label 

train["Survived"] = train["Survived"].astype(int)
Y_train = train["Survived"]
X_train = train.drop(labels = ["Survived"],axis = 1)        
        
#%%
# 6.1 Simple Model

# 6.1.1 Cross Validation Models

# ------------------------ #
#SVC
#Decision Tree
#AdaBoost
#Random Forest
#Extra Trees
#Gradient Boosting
#Multiple layer perceprton (neural network)
#KNN
#Logistic regression
#Linear Discriminant Analysis
# ------------------------- #

# Cross validate model with Kfold stratified cross val
kfold = StratifiedKFold(n_splits=10)

# Modeling step Test differents algorithms 
random_state = 2
classifiers = []
classifiers.append(SVC(random_state=random_state))
classifiers.append(DecisionTreeClassifier(random_state=random_state))
classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=0.1))
classifiers.append(RandomForestClassifier(random_state=random_state))
classifiers.append(ExtraTreesClassifier(random_state=random_state))
classifiers.append(GradientBoostingClassifier(random_state=random_state))
classifiers.append(MLPClassifier(random_state=random_state))
classifiers.append(KNeighborsClassifier())
classifiers.append(LogisticRegression(random_state = random_state))
classifiers.append(LinearDiscriminantAnalysis())

cv_results = []
for classifier in classifiers :
    cv_results.append(cross_val_score(classifier, X_train, 
                                      y = Y_train, scoring = "accuracy", 
                                      cv = kfold, n_jobs=-1))
    
cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())

cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,
                       "Algorithm":["SVC","DecisionTree","AdaBoost",
"RandomForest","ExtraTrees","GradientBoosting","MultipleLayerPerceptron",
"KNeighboors","LogisticRegression","LinearDiscriminantAnalysis"]})

g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, 
                palette="Set3",orient = "h",**{'xerr':cv_std})
g.set_xlabel("Mean Accuracy")
g = g.set_title("Cross validation scores")

# I decided to choose the SVC, AdaBoost, RandomForest , ExtraTrees and 
# the GradientBoosting classifiers for the ensemble modeling.

#%%        
# 6.1.2 Hyperparameter tunning for best model  
 
# grid search optimization for AdaBoost, ExtraTrees , RandomForest, 
# GradientBoosting and SVC classifiers.    

#%%
# Adaboost
DTC = DecisionTreeClassifier()
adaDTC = AdaBoostClassifier(DTC, random_state=7)
ada_param_grid = {"base_estimator__criterion" : ["gini", "entropy"],
              "base_estimator__splitter" :   ["best", "random"],
              "algorithm" : ["SAMME","SAMME.R"],
              "n_estimators" :[1,2],
              "learning_rate":  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,1.5]}

gsadaDTC = GridSearchCV(adaDTC,param_grid = ada_param_grid, 
                        cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsadaDTC.fit(X_train,Y_train)

ada_best = gsadaDTC.best_estimator_
print(gsadaDTC.best_score_)
# 0.82406356413166859

#%%
#ExtraTrees 
ExtC = ExtraTreesClassifier()


## Search grid for optimal parameters
ex_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}


gsExtC = GridSearchCV(ExtC,param_grid = ex_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsExtC.fit(X_train,Y_train)

ExtC_best = gsExtC.best_estimator_

# Best score
gsExtC.best_score_
# 0.82973893303064694

#%%
 RFC Parameters tunning 
RFC = RandomForestClassifier()


## Search grid for optimal parameters
rf_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}


gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsRFC.fit(X_train,Y_train)

RFC_best = gsRFC.best_estimator_

# Best score
gsRFC.best_score_
# 0.83427922814982969

#%%
# Gradient boosting tunning

GBC = GradientBoostingClassifier()
gb_param_grid = {'loss' : ["deviance"],
              'n_estimators' : [100,200,300],
              'learning_rate': [0.1, 0.05, 0.01],
              'max_depth': [4, 8],
              'min_samples_leaf': [100,150],
              'max_features': [0.3, 0.1] 
              }

gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsGBC.fit(X_train,Y_train)

GBC_best = gsGBC.best_estimator_

# Best score
gsGBC.best_score_
# 0.83087400681044266

#%%
### SVC classifier
SVMC = SVC(probability=True)
svc_param_grid = {'kernel': ['rbf'], 
                  'gamma': [ 0.001, 0.01, 0.1, 1],
                  'C': [1, 10, 50, 100,200,300, 1000]}

gsSVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsSVMC.fit(X_train,Y_train)

SVMC_best = gsSVMC.best_estimator_

# Best score
gsSVMC.best_score_
# 0.83314415437003408

#%%
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    """Generate a simple plot of the test and training learning curve"""
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

g = plot_learning_curve(gsRFC.best_estimator_,"RF mearning curves",X_train,Y_train,cv=kfold)
g = plot_learning_curve(gsExtC.best_estimator_,"ExtraTrees learning curves",X_train,Y_train,cv=kfold)
g = plot_learning_curve(gsSVMC.best_estimator_,"SVC learning curves",X_train,Y_train,cv=kfold)
g = plot_learning_curve(gsadaDTC.best_estimator_,"AdaBoost learning curves",X_train,Y_train,cv=kfold)
g = plot_learning_curve(gsGBC.best_estimator_,"GradientBoosting learning curves",X_train,Y_train,cv=kfold)
        
#%%
# 6.1.4 Feature importance of tree based classifiers¶

nrows = ncols = 2
fig, axes = plt.subplots(nrows = nrows, ncols = ncols, sharex="all", figsize=(15,15))

names_classifiers = [("AdaBoosting", ada_best),("ExtraTrees",ExtC_best),("RandomForest",RFC_best),("GradientBoosting",GBC_best)]

nclassifier = 0
for row in range(nrows):
    for col in range(ncols):
        name = names_classifiers[nclassifier][0]
        classifier = names_classifiers[nclassifier][1]
        indices = np.argsort(classifier.feature_importances_)[::-1][:40]
        g = sns.barplot(y=X_train.columns[indices][:40],x = classifier.feature_importances_[indices][:40] , orient='h',ax=axes[row][col])
        g.set_xlabel("Relative importance",fontsize=12)
        g.set_ylabel("Features",fontsize=12)
        g.tick_params(labelsize=9)
        g.set_title(name + " feature importance")
        nclassifier += 1       
        
#%%
test_Survived_RFC = pd.Series(RFC_best.predict(test), name="RFC")
test_Survived_ExtC = pd.Series(ExtC_best.predict(test), name="ExtC")
test_Survived_SVMC = pd.Series(SVMC_best.predict(test), name="SVC")
test_Survived_AdaC = pd.Series(ada_best.predict(test), name="Ada")
test_Survived_GBC = pd.Series(GBC_best.predict(test), name="GBC")


# Concatenate all classifier results
ensemble_results = pd.concat([test_Survived_RFC,test_Survived_ExtC,test_Survived_AdaC,test_Survived_GBC, test_Survived_SVMC],axis=1)


g= sns.heatmap(ensemble_results.corr(),annot=True)

#%%
# 6.2 Ensemble modeling
votingC = VotingClassifier(estimators=[('rfc', RFC_best), ('extc', ExtC_best),
('svc', SVMC_best), ('adac',ada_best),('gbc',GBC_best)], voting='soft', n_jobs=4)

votingC = votingC.fit(X_train, Y_train)

#%%
# 6.3 Prediction
# 6.3.1 Predict and Submit results    

test_Survived = pd.Series(votingC.predict(test), name="Survived")

results = pd.concat([IDtest,test_Survived],axis=1)

results.to_csv("ensemble_python_voting.csv",index=False)

    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        