# -*- coding: utf-8 -*-
import os
from random import shuffle
from nltk import wordpunct_tokenize
import ujson

"""
We're going to need to have the data set from "http://ai.stanford.edu/~amaas/data/sentiment/"

When we extract aclImdb_v1.tar.gz it produces a folder called "aclImdb"
    that folder needs to be in the same directory as imdb_example.py and this file prep_data.py

Run this, and we should be able to run the example.
"""

train_doc_data = []

f_gen = os.walk('aclImdb/train/pos')

for fitem in f_gen:
    for fname in fitem[2]:
        if fname.endswith('.txt'):
            train_doc_data.append(ujson.dumps((wordpunct_tokenize(open(os.path.join('aclImdb/train/pos', fname)).read().lower().replace('<br />', 'lbr ')), ['αþ' + fname[:-4]])) + '\n')

f_gen = os.walk('aclImdb/train/neg')

for fitem in f_gen:
    for fname in fitem[2]:
        if fname.endswith('.txt'):
            train_doc_data.append(ujson.dumps((wordpunct_tokenize(open(os.path.join('aclImdb/train/neg', fname)).read().lower().replace('<br />', 'lbr ')), ['αñ' + fname[:-4]])) + '\n')


f_gen = os.walk('aclImdb/train/unsup')

for fitem in f_gen:
    for fname in fitem[2]:
        if fname.endswith('.txt'):
            train_doc_data.append(ujson.dumps((wordpunct_tokenize(open(os.path.join('aclImdb/train/unsup', fname)).read().lower().replace('<br />', 'lbr ')), ['αû' + fname[:-4]])) + '\n')

shuffle(train_doc_data)

with open('train_data.csv', 'w') as trd_f:
    trd_f.writelines(train_doc_data)


test_doc_data = []

f_gen = os.walk('aclImdb/test/pos')

for fitem in f_gen:
    for fname in fitem[2]:
        if fname.endswith('.txt'):
            test_doc_data.append(ujson.dumps((wordpunct_tokenize(open(os.path.join('aclImdb/test/pos', fname)).read().replace('<br />', 'lbr')), ['βþ' + fname[:-4]])) + '\n')

f_gen = os.walk('aclImdb/test/neg')

for fitem in f_gen:
    for fname in fitem[2]:
        if fname.endswith('.txt'):
            test_doc_data.append(ujson.dumps((wordpunct_tokenize(open(os.path.join('aclImdb/test/neg', fname)).read().replace('<br />', 'lbr')), ['βñ' + fname[:-4]])) + '\n')


shuffle(test_doc_data)

with open('test_data.csv', 'w') as ted_f:
    ted_f.writelines(test_doc_data)