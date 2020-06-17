#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 10:48:06 2020

@author: ernestng
"""


# import data and set column names
import pandas as pd

df = pd.read_csv('./spam.csv', encoding="latin-1")
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
df.columns = ['label', 'text']

# remove special characters
import re

def clean_text(df, text_field, new_text_field_name):
    df[new_text_field_name] = df[text_field].str.lower()
    df[new_text_field_name] = df[new_text_field_name].apply(lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem))  
    # remove numbers
    df[new_text_field_name] = df[new_text_field_name].apply(lambda elem: re.sub(r"\d+", "", elem))
    
    return df

df = clean_text(df, "text", "text_clean")

# remove stopwords
import nltk.corpus
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')
stop.extend(['the', 'that', 'but', 'they', 'and', 'you', 'we', 'he', 'this', 'not', 'what', 'so',
             'I\'m', 'are', 'if', 'to', 'said', 'say', 'don\'t','u','ur','im','ok','ill'])

df['text_clean'] = df['text_clean'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

# lemmatization
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

def word_lemmatizer(text):
    lem_text = [WordNetLemmatizer().lemmatize(i) for i in text]
    return lem_text


df['text_tokens'] = df['text_clean'].apply(lambda x: word_tokenize(x))

df['text_tokens_lemma'] = df['text_tokens'].apply(lambda x: word_lemmatizer(x))
df['text_tokens_lemma'].head()

df['text'] = df['text_tokens_lemma'].apply(lambda x: ' '.join(x))

#FEATURE ENGINEERING
#Get word count
df['word_count'] = df['text'].apply(lambda x: len(str(x.split())))    

#Part of Speech Counts

from textblob import TextBlob

pos_family = {
    'noun' : ['NN','NNS','NNP','NNPS'],
    'pron' : ['PRP','PRP$','WP','WP$'],
    'verb' : ['VB','VBD','VBG','VBN','VBP','VBZ'],
    'adj' :  ['JJ','JJR','JJS'],
    'adv' : ['RB','RBR','RBS','WRB']
}

# function to check and get the part of speech tag count of a words in a given sentence
def check_pos_tag(x, flag):
    cnt = 0
    try:
        wiki = TextBlob(x)
        for tup in wiki.tags:
            ppo = list(tup)[1]
            if ppo in pos_family[flag]:
                cnt += 1
    except:
        pass
    return cnt

df['noun_count'] = df['text'].apply(lambda x: check_pos_tag(x, 'noun'))
df['verb_count'] = df['text'].apply(lambda x: check_pos_tag(x, 'verb'))
df['adj_count'] = df['text'].apply(lambda x: check_pos_tag(x, 'adj'))
df['adv_count'] = df['text'].apply(lambda x: check_pos_tag(x, 'adv'))
df['pron_count'] = df['text'].apply(lambda x: check_pos_tag(x, 'pron'))

#save dataframe
df.to_csv('./model_data.csv')