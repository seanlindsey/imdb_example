# -*- coding: utf-8 -*-
import unicodedata

from urllib.parse import unquote as unquote_url
from html import unescape

from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.tokenize import wordpunct_tokenize
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


class BOWObject(object):
    def __init__(self, fname):
        self.fobj = open(fname)
        self.train = {}
        self.test = {}
        # A few options here
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

