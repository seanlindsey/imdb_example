# -*- coding: utf-8 -*-
import os
from random import shuffle
from nltk import wordpunct_tokenize
import ujson
from six import text_type

"""
We're going to need to have the data set from "http://ai.stanford.edu/~amaas/data/sentiment/"

When we extract aclImdb_v1.tar.gz it produces a folder called "aclImdb"
    that folder needs to be in the same directory as imdb_example.py and this file prep_data.py

Run this, and we should be able to run the example.
"""

train_doc_data = []

proc_fname = lambda xstr, xfname, xlab: text_type(ujson.dumps(
    (
        wordpunct_tokenize(
            text_type(xstr)#.lower().replace(u'<br />', u'lbr ')
        ), [xlab + text_type(xfname[:-4])]
    )) + u'\n')

get_str_iterator = lambda xstr: [xstr]

f_gen = os.walk('aclImdb/train/pos')

for fitem in f_gen:
    for fname in fitem[2]:
        if fname.endswith('.txt'):
            doc_string = open(os.path.join('aclImdb/train/pos', fname), 'rb').read().decode(encoding='utf-8')
            doc_string_list = get_str_iterator(doc_string)
            for some_str in doc_string_list:
                train_doc_data.append(proc_fname(some_str, fname, u'αþ'))


f_gen = os.walk('aclImdb/train/neg')

for fitem in f_gen:
    for fname in fitem[2]:
        if fname.endswith('.txt'):
            doc_string = open(os.path.join('aclImdb/train/neg', fname), 'rb').read().decode(encoding='utf-8')
            doc_string_list = get_str_iterator(doc_string)
            for some_str in doc_string_list:
                train_doc_data.append(proc_fname(some_str, fname, u'αñ'))
            #train_doc_data.append(ujson.dumps((wordpunct_tokenize(open(os.path.join('aclImdb/train/neg', fname)).read().lower().replace('<br />', 'lbr ')), ['αñ' + fname[:-4]])) + '\n')


f_gen = os.walk('aclImdb/train/unsup')

for fitem in f_gen:
    for fname in fitem[2]:
        if fname.endswith('.txt'):
            doc_string = open(os.path.join('aclImdb/train/unsup', fname), 'rb').read().decode(encoding='utf-8')
            doc_string_list = get_str_iterator(doc_string)
            for some_str in doc_string_list:
                train_doc_data.append(proc_fname(some_str, fname, u'αû'))
            #train_doc_data.append(ujson.dumps((wordpunct_tokenize(open(os.path.join('aclImdb/train/unsup', fname)).read().lower().replace('<br />', 'lbr ')), ['αû' + fname[:-4]])) + '\n')

shuffle(train_doc_data)

with open('train_data.csv', 'w') as trd_f:
    trd_f.writelines(train_doc_data)


test_doc_data = []

f_gen = os.walk('aclImdb/test/pos')

for fitem in f_gen:
    for fname in fitem[2]:
        if fname.endswith('.txt'):
            doc_string = open(os.path.join('aclImdb/test/pos', fname), 'rb').read().decode(encoding='utf-8')
            doc_string_list = get_str_iterator(doc_string)
            for some_str in doc_string_list:
                test_doc_data.append(proc_fname(some_str, fname, u'βþ'))
            #test_doc_data.append(ujson.dumps((wordpunct_tokenize(open(os.path.join('aclImdb/test/pos', fname)).read().replace('<br />', 'lbr')), ['βþ' + fname[:-4]])) + '\n')

f_gen = os.walk('aclImdb/test/neg')

for fitem in f_gen:
    for fname in fitem[2]:
        if fname.endswith('.txt'):
            doc_string = open(os.path.join('aclImdb/test/neg', fname), 'rb').read().decode(encoding='utf-8')
            doc_string_list = get_str_iterator(doc_string)
            for some_str in doc_string_list:
                test_doc_data.append(proc_fname(some_str, fname, u'βñ'))
            # test_doc_data.append(ujson.dumps((wordpunct_tokenize(open(os.path.join('aclImdb/test/neg', fname)).read().replace('<br />', 'lbr')), ['βñ' + fname[:-4]])) + '\n')


shuffle(test_doc_data)

with open('test_data.csv', 'w') as ted_f:
    ted_f.writelines(test_doc_data)