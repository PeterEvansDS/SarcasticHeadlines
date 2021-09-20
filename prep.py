#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 11:34:47 2021


"""
import string
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import spacy
from collections import Counter
import plotly.express as px

df = pd.read_csv('./df.csv', converters={'lemmas': eval, 'entities':eval, 
                                         'entity labels':eval, 'label definitions':eval})
sarcastic = df.loc[df['is_sarcastic']==1]
non = df.loc[df['is_sarcastic']!=1]
nlp = spacy.load("en_core_web_sm")


#most popular lemmas
s_lemmas = pd.DataFrame(sarcastic.lemmas.explode().value_counts())
n_lemmas = pd.DataFrame(non.lemmas.explode().value_counts())
s_lemmas['is_sarcastic'] = 1
n_lemmas['is_sarcastic'] = 0
lemmas = pd.concat([s_lemmas, n_lemmas], axis=0)


#most popular entities
s_entities = pd.DataFrame({'entities':sarcastic.entities.explode(),
                           'entity labels':sarcastic['entity labels'].explode(),
                           'label definitions':sarcastic['label definitions'].explode()}).value_counts().iloc[1:].reset_index()
n_entities = pd.DataFrame({'entities':non.entities.explode(),
                           'entity labels':non['entity labels'].explode(),
                           'label definitions':non['label definitions'].explode()}).value_counts().iloc[1:].reset_index()

s_entities['Article Type'] = 'Sarcastic'
n_entities['Article Type'] = 'Legitimate'
entities = pd.concat([s_entities, n_entities], axis=0)
entities.columns = ['entities', 'entity labels', 'label definitions', 'number of instances', 'article type']
print(entities['entity labels'].unique())
vc = entities[['entity labels', 'label definitions']].value_counts().reset_index()
vc.to_csv('./vc.csv')

lemmas.to_csv('./lemmas.csv')
entities.to_csv('./entities1.csv')




