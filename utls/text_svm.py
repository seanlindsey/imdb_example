# -*- coding: utf-8 -*-
import csv
import random
import unicodedata
import numpy as np

from urllib.parse import unquote as unquote_url
from html import unescape

from sklearn.datasets.base import Bunch
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.svm import SVC
from sklearn import metrics

from nltk.tokenize import wordpunct_tokenize
# from nltk.stem.snowball import EnglishStemmer
from nltk.corpus import stopwords
from six.moves import xrange
import ujson


PUNC_DICT = {
    '!': ' exclamationpoint ',
    '"': ' doublequote ',
    '#': ' numbersign ',
    '$': ' dollarsign ',
    '%': ' percentsign ',
    '&': ' amp ',
    '(': ' openparenthesis ',
    ')': ' closeparenthesis ',
    '*': ' astrixsign ',
    '+': ' plussign ',
    ',': ' commasign ',
    '.': ' periodmark ',
    '/': ' fowardslash ',
    ':': ' fullcolon ',
    ';': ' semicolon ',
    '<': ' lessthan ',
    '=': ' equalsign ',
    '>': ' greaterthan ',
    '?': ' questionmark ',
    '@': ' atsign ',
    '[': ' openbracket ',
    '\\': ' backslash ',
    ']': ' closebracket ',
    '^': ' carrotsign ',
    '_': ' underscore ',
    '{': ' opencurly ',
    '|': ' pipebar ',
    '}': ' closecurly ',
    '~': ' tildasign ',
}

WORD_PUNC_LIST = [
    'exclamationpoint',
    'doublequote',
    'numbersign',
    'dollarsign',
    'percentsign',
    'amp',
    'openparenthesis',
    'closeparenthesis',
    'astrixsign',
    'plussign',
    'commasign',
    'periodmark',
    'fowardslash',
    'fullcolon',
    'semicolon',
    'lessthan',
    'equalsign',
    'greaterthan',
    'questionmark',
    'atsign',
    'openbracket',
    'backslash',
    'closebracket',
    'carrotsign',
    'underscore',
    'opencurly',
    'pipebar',
    'closecurly',
    'tildasign',
]


class TextSVM(SVC):
#    svm_ = SVC(C=500, kernel='poly', gamma=.01, shrinking=True, probability=False, degree= 10, coef0=2,
#        tol=0.001, cache_size=20000, class_weight=None, verbose=False, max_iter=-1)
    def __init__(self, train_data, C=5, kernel='poly', gamma=.001, degree=10, coef0=2, n_features=10000000,
                 ngram_range=(1, 10), tfidf=False, dfrange=(2, 1.0), probability=False, class_weight=None):
        self.score = None
        self.accuracy = None
        self.is_tfidf = tfidf
        if tfidf:
            self.vectorizer = TfidfVectorizer(stop_words=None, min_df=dfrange[0], max_df=dfrange[1],
                                              max_features=n_features, strip_accents='unicode',
                                              ngram_range=ngram_range, analyzer='word', norm='l2')
        else:
            self.vectorizer = HashingVectorizer(stop_words=None, non_negative=True,
                                                n_features=n_features, strip_accents='unicode',
                                                ngram_range=ngram_range, analyzer='word', norm='l2')
        self.param_set = {'C': str(C), 'kernel': str(kernel), 'gamma': str(gamma),
                          'degree': str(degree), 'coef0': str(coef0), 'n_features': str(n_features)}
        if class_weight == 'auto':
            class_weight = {}
            for item in train_data.target:
                if class_weight.get(item):
                    class_weight.update({item: class_weight[item] + 1.0})
                else:
                    class_weight.update({item: 1.0})
            for key in class_weight:
                class_weight.update({key: 1.0 / class_weight[key]})
        self.class_weight_dict = class_weight
        super(TextSVM, self).__init__(C=C, kernel=kernel, gamma=gamma, shrinking=True, probability=probability, degree=degree, coef0=coef0,
                                      tol=0.001, cache_size=20000, class_weight=class_weight, verbose=False, max_iter=-1)
        if self.is_tfidf:
            train_x = self.vectorizer.fit_transform(train_data.data)
        else:
            train_x = self.vectorizer.transform(train_data.data)
        self.fit(train_x, train_data.target)

    def test_data(self, test_data):
        test_x = self.vectorizer.transform(test_data.data)
        predicted_values = self.predict(test_x)
        test_y = test_data.target
        self.score = metrics.f1_score(test_y, predicted_values)
        self.accuracy = metrics.accuracy_score(test_y, predicted_values)

    def guess_text(self, text_text):
        text_x = self.vectorizer.transform([pre_proc(text_text, removestop=True, word_punc=False, unquote=True), ])
        return self.predict(text_x)


def pre_proc(in_str, removestop=True, word_punc=False, unquote=False):
    # remove accents, wordify punctuation
    in_str = strip_accents(in_str, wordify=word_punc, unquote=unquote)
    # tokenize string
    if removestop:  # remove stop words
        tok_list = filter(lambda x: x not in stopwords.words('english'), wordpunct_tokenize(in_str))
    else:
        tok_list = wordpunct_tokenize(in_str)
    out_str = ' '.join(tok_list)
    return out_str.lower()


def strip_accents(s, wordify=False, unquote=False):
    if unquote:
        s = unquote_url(s)
        s = unescape(s)
    if wordify:
        pre_rv = ''.join((c if c not in '!"#$%&()*+,./:;<=>?@[\\]^_{|}~' else PUNC_DICT[c]
                          for c in unicodedata.normalize('NFD', s)
                          if unicodedata.category(c) != 'Mn' and c not in '\'`'))
    else:
        pre_rv = ''.join((c if c not in '!"#$%&()*+,./:;<=>?@[\\]^_{|}~' else ' '
                          for c in unicodedata.normalize('NFD', s)
                          if unicodedata.category(c) != 'Mn' and c not in '\'`'))
    rv = ''
    for c in pre_rv:
        if repr(c).__contains__(r'\x'):
            rv += repr(c)[2:-1]
        else:
            rv += c
    return rv


def bunch_data(data_bundle, fav_number=None):
    new_data = []
    new_targets = []
    for i in range(len(data_bundle.data)):
        if data_bundle.data[i] not in new_data:
            new_data.append(data_bundle.data[i])
            if fav_number is None:
                new_targets.append((data_bundle.target[i],))
            else:
                new_targets.append(data_bundle.target[i])
        else:
            for j in range(len(new_data)):
                if data_bundle.data[i] == new_data[j] and fav_number is None:
                    new_targets[j] += (data_bundle.target[i],)
                elif data_bundle.data[i] == new_data[j] and \
                        abs(float(data_bundle.target[i]) - fav_number) < abs(float(new_targets[j]) - fav_number):
                    new_targets[j] = data_bundle.target[i]
    return Bunch(data=new_data, target=np.array(new_targets, dtype=np.float64), target_names=[])


class BOWObject(object):
    def __init__(self, fname):
        self.fobj = open(fname)
        self.train = {}
        self.test = {}
        self.vectorizer = TfidfVectorizer(stop_words=None, min_df=2, max_df=.95,
                                          max_features=10000000, strip_accents='unicode',
                                          ngram_range=(1, 2), analyzer='word', norm='l2')

    def build_train(self):
        l_texts = []
        l_lbls = []
        l_ids = []
        for line in self.fobj:
            l_text, l_label = ujson.loads(line[:-1] if line.endswith('\n') else line)
            l_texts.append(l_text)
            l_ids.append(l_label[0])
            if l_label[0][1] == 'þ':
                l_lbls.append(1)
            elif l_label[0][1] == 'ñ':
                l_lbls.append(0)
        train_x = self.vectorizer.fit_transform(l_texts)
        idx_gen = xrange(len(l_ids)).__iter__()
        for tfidf_matrix in train_x:
            tfidf_matrix_idx = next(idx_gen)
            self.train.update({l_ids[tfidf_matrix_idx]: (l_lbls[tfidf_matrix_idx], tfidf_matrix)})

    def build_test(self, fname):
        test_fobj = open(fname)
        l_texts = []
        l_lbls = []
        l_ids = []
        for line in test_fobj:
            l_text, l_label = ujson.loads(line[:-1] if line.endswith('\n') else line)
            l_texts.append(l_text)
            l_ids.append(l_label[0])
            if l_label[0][1] == 'þ':
                l_lbls.append(1)
            elif l_label[0][1] == 'ñ':
                l_lbls.append(0)
        train_x = self.vectorizer.transform(l_texts)
        idx_gen = xrange(len(l_ids)).__iter__()
        for tfidf_matrix in train_x:
            tfidf_matrix_idx = next(idx_gen)
            self.test.update({l_ids[tfidf_matrix_idx]: (l_lbls[tfidf_matrix_idx], tfidf_matrix)})



def fetch_data_bow(file_to_open):
    the_targets = []
    the_data = []
    for line in open(file_to_open):
        l_data, l_label = ujson.loads(line[:-1] if line.endswith('\n') else line)

        the_targets.append(np.int())


    # print_count = 0
    # rcsv = csv.reader(open(csv_to_open, 'r'))
    # csv_lbls_2 = []
    # csv_texts_2 = []
    # first_row = True
    # rcsv_list = []
    # for row in rcsv:
    #     if not first_row:
    #         rcsv_list.append(row)
    #     first_row = False
    # random.shuffle(rcsv_list)
    # for row in rcsv_list:
    #     csv_lbls_2.append(int(row[0]))
    #     text_row = pre_proc(row[1], removestop=False, alwayskeep=True, word_punc=True, unquote=True)
    #     csv_texts_2.append(text_row)
    #     print_count += 1
    #     print(print_count, '::: ', row[1], ' --> ', text_row)
    # split_idx = int(len(rcsv_list)*.1)
    # rv = Bunch(data=csv_texts_2, target=np.array(csv_lbls_2[:split_idx], dtype=np.float64), target_names=[])
    # return rv