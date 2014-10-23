# -*- coding: utf-8 -*-
import os
from random import shuffle
from nltk import wordpunct_tokenize
import ujson
from utls.text_svm import pre_proc

"""
We're going to need to have the data set from "http://ai.stanford.edu/~amaas/data/sentiment/"

When we extract aclImdb_v1.tar.gz it produces a folder called "aclImdb"
    that folder needs to be in the same directory as imdb_example.py and this file prep_data.py

Run this, and we should be able to run the example.
"""

train_path_info = [('aclImdb/train/pos', 'αþ'), ('aclImdb/train/neg', 'αñ'), ('aclImdb/train/unsup', 'αû')]
test_path_info = [('aclImdb/test/pos', 'βþ'), ('aclImdb/test/neg', 'βñ')]

proc_data_lti = lambda x, y: ujson.dumps((wordpunct_tokenize(
    open(os.path.join(x, fname)).read().lower().replace('<br />', 'lbr ')
), [y + fname[:-4]])) + '\n'

proc_data_bow = lambda x, y: ujson.dumps((pre_proc(
    open(os.path.join(x, fname)).read(), removestop=True, word_punc=False, unquote=True
), [y + fname[:-4]])) + '\n'


####  --- data for LabeledText --- ####

train_doc_data = []
test_doc_data = []

for path_info in train_path_info:
    f_gen = os.walk(path_info[0])
    for fitem in f_gen:
        for fname in fitem[2]:
            if fname.endswith('.txt'):
                train_doc_data.append(proc_data_lti(path_info[0], path_info[1]))

shuffle(train_doc_data)

with open('train_data.csv', 'w') as trd_f:
    trd_f.writelines(train_doc_data)

print('done train_data.csv')

for path_info in test_path_info:
    f_gen = os.walk(path_info[0])
    for fitem in f_gen:
        for fname in fitem[2]:
            if fname.endswith('.txt'):
                test_doc_data.append(proc_data_lti(path_info[0], path_info[1]))

shuffle(test_doc_data)

with open('test_data.csv', 'w') as ted_f:
    ted_f.writelines(test_doc_data)

print('done test_data.csv')

####  --- data for BOWs --- ####

train_bow_data = []
test_bow_data = []

for path_info in train_path_info[:2]:
    f_gen = os.walk(path_info[0])
    for fitem in f_gen:
        for fname in fitem[2]:
            if fname.endswith('.txt'):
                train_bow_data.append(proc_data_bow(path_info[0], path_info[1]))

shuffle(train_bow_data)

with open('train_data_bow.csv', 'w') as trd_f:
    trd_f.writelines(train_bow_data)

print('done train_data_bow.csv')

for path_info in test_path_info:
    f_gen = os.walk(path_info[0])
    for fitem in f_gen:
        for fname in fitem[2]:
            if fname.endswith('.txt'):
                test_bow_data.append(proc_data_bow(path_info[0], path_info[1]))

shuffle(test_bow_data)

with open('test_data_bow.csv', 'w') as ted_f:
    ted_f.writelines(test_bow_data)

print('done test_data_bow.csv')