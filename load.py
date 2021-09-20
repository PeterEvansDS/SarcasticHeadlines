import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import spacy
from collections import Counter

def parse_data(file):
    for l in open(file,'r'):
        yield json.loads(l)

data = list(parse_data('./Sarcasm_Headlines_Dataset_v2.json'))
df = pd.DataFrame.from_dict(data, orient='columns')

#number of tokens (units of text)
nlp = spacy.load("en_core_web_sm")
df['doc'] = df.headline.apply(lambda x: nlp(x))
df['headline length'] = df.doc.apply(lambda x: len((x)))
df['entities'] = df.doc.apply(lambda x: [entity.text for entity in x.ents])
df['entity labels'] = df.doc.apply(lambda x: [entity.label_ for entity in x.ents])
df['label definitions'] = df['entity labels'].apply(lambda x: [str(spacy.explain(label)) for label in x ])
df['parts of speech'] = df.doc.apply(lambda x: [token.pos_ for token in x])
df['doc w/o stops'] = df.doc.apply(lambda x: [word for word in x if word.is_stop == False])
df['lemmas'] = df['doc w/o stops'].apply(lambda x: [token.lemma_ for token in x])

#count individudal items
dicts = df['parts of speech'].apply(lambda x: Counter(x)).apply(pd.Series)
dicts = dicts.fillna(0)
ents = df['entity labels'].apply(lambda x: Counter(x)).apply(pd.Series)
ents = ents.fillna(0)

df = df[['is_sarcastic', 'headline', 'doc', 'doc w/o stops', 'lemmas', 'headline length',
         'entities', 'entity labels', 'label definitions']]
df = pd.concat([df, dicts, ents], axis=1)

df.to_csv('./df.csv')

print('NUMBER OF SARCASTIC VALUES: \n', df.is_sarcastic.value_counts())
print('TOTAL NUMBER OF NULL VALUES: ', df.isnull().sum().sum())
duplicates = df.astype(str).duplicated()
print('NUMBER OF DUPLICATE VALUES: ', df.astype(str).duplicated().sum())
print('DUPLICATES: \n', df[df.astype(str).duplicated(keep=False).values].headline)
df = df.astype(str).drop_duplicates()

df.to_csv('./df.csv')
