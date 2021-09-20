#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 12:43:31 2021

@author: peterevans
"""

import string
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import spacy
from collections import Counter
import plotly.express as px
import time

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score

#build the bag of words features

start = time.time()
df = pd.read_csv('./df.csv', converters={'lemmas': eval, 'entities':eval, 
                                         'entity labels':eval, 'label definitions':eval})
bow = df.lemmas.apply(lambda x: Counter(x)).apply(pd.Series)
bow = bow.fillna(0)
bow.to_feather('./bow.feather')
end = time.time()
print('Runtime: ', end-start)


#%%build a simple bag of words model and test with a few different classification algorithms

def getResults(model, name, results, X_train, y_train, X_test, y_test):
    cv_results = cross_validate(model, X = X_train, y=y_train, scoring=('accuracy'), return_train_score=True, n_jobs=-1)
    fit_start = time.time()
    model.fit(X_train, y_train)
    fit_finish = time.time()
    
    pred_start = time.time()
    y_pred = model.predict(X_test)
    pred_finish = time.time()

    results = results.append({'Model': name,
                    'Training Accuracy':np.mean(cv_results['train_score']),
                    'Validation Accuracy':np.mean(cv_results['test_score']),
                    'Test Accuracy':accuracy_score(y_test, y_pred),
                    'Build Time': fit_finish-fit_start,
                    'Classification Time': pred_finish-pred_start}, ignore_index=True)
    print('DONE: ', name)
    return results
    
results = pd.DataFrame(columns=['Model', 'Training Accuracy', 'Validation Accuracy', 'Test Accuracy', 'Build Time',
                                'Classification Time'])
#basic decision tree
X = bow
y = df['is_sarcastic']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model = DecisionTreeClassifier()
results = getResults(model, 'DecisionTreeBOW', results,  X_train, y_train, X_test, y_test)

#decision tree with only extra text analysis data
X = pd.concat([df.loc[:, 'VERB':'LAW'], df['headline length']], axis=1)
y = df['is_sarcastic']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model = DecisionTreeClassifier()
results = getResults(model, 'DecisionTreeDF', results,  X_train, y_train, X_test, y_test)

#decision tree with both
X = pd.concat([bow, X], axis=1)
y = df['is_sarcastic']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model = DecisionTreeClassifier()
results = getResults(model, 'DecisionTreeBoth', results,  X_train, y_train, X_test, y_test)

results.to_csv('./bowresults.csv')

#%% random forests

from sklearn.ensemble import RandomForestClassifier

#basic random forest
X = bow
y = df['is_sarcastic']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model = RandomForestClassifier()
results = getResults(model, 'RandomForestBOW', results,  X_train, y_train, X_test, y_test)

X = pd.concat([df.loc[:, 'VERB':'LAW'], df['headline length']], axis=1)
y = df['is_sarcastic']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
results = getResults(model, 'RandomForestDF', results,  X_train, y_train, X_test, y_test)

X = pd.concat([bow, X], axis=1)
y = df['is_sarcastic']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
results = getResults(model, 'RandomForestBoth', results,  X_train, y_train, X_test, y_test)

results.to_csv('./bowresults.csv')

#%% support vector machines
#basic random forest
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier

def getResults(model, name, results, X_train, y_train, X_test, y_test):
    cv_results = cross_validate(model, X = X_train, y=y_train, scoring=('accuracy'), return_train_score=True)
    fit_start = time.time()
    model.fit(X_train, y_train)
    fit_finish = time.time()
    
    pred_start = time.time()
    y_pred = model.predict(X_test)
    pred_finish = time.time()

    results = results.append({'Model': name,
                    'Training Accuracy':np.mean(cv_results['train_score']),
                    'Validation Accuracy':np.mean(cv_results['test_score']),
                    'Test Accuracy':accuracy_score(y_test, y_pred),
                    'Build Time': fit_finish-fit_start,
                    'Classification Time': pred_finish-pred_start}, ignore_index=True)
    print('DONE: ', name)
    return results

results = pd.DataFrame(columns=['Model', 'Training Accuracy', 'Validation Accuracy', 'Test Accuracy', 'Build Time',
                                'Classification Time'])
    
n_estimators = 20

X = bow
y = df['is_sarcastic']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model = BaggingClassifier(SVC(), max_samples=1.0 / n_estimators, n_estimators=n_estimators, n_jobs=-1)
results = getResults(model, 'SVCBOW', results,  X_train, y_train, X_test, y_test)

X = pd.concat([df.loc[:, 'VERB':'LAW'], df['headline length']], axis=1)
y = df['is_sarcastic']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
results = getResults(model, 'SVCDF', results,  X_train, y_train, X_test, y_test)

X = pd.concat([bow, X], axis=1)
y = df['is_sarcastic']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
results = getResults(model, 'SVCBoth', results,  X_train, y_train, X_test, y_test)

results.to_csv('./bowresults_svc.csv')




