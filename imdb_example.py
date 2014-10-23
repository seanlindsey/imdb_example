# -*- coding: utf-8 -*-
from collections import namedtuple

# From package scikit-learn
from sklearn.svm import SVC
from sklearn.metrics import metrics
from sklearn.linear_model import SGDClassifier

from utls.imdb_example_utils import LTIterator, add_labeled_texts

from numpy import concatenate
from gensim.models.doc2vec import Doc2Vec
from numpy import array, int as np_int
from six import iteritems
from six.moves import xrange
import logging

# see paper "Distributed Representations of Sentences and Documents"
#   http://cs.stanford.edu/~quocle/paragraph_vector.pdf
# Authors: Quoc Le, Tomas Mikolov

print('An attempt to reproduce something like "Experiment 3.2: IMDB sentiment", from the paper.\n'
      '\tThis is gonna take a while.')

VEC_SIZE = 400
WINDOW_SIZE = 10
MIN_COUNT = 9
# A note about logging with doc2vec: We are going to need to come up with our own word counts to get accurate percentages
#   An accurate count could come from counting everything in the vocab that's not a label if training on the full set.
#   Also anything could return an accurate count that passes over our iterator.
OUTPUT_OVERLOAD = False

TEST_GET_VEC = False
TRY_WITH_PREBUILD = False

print('Building the 75k training vectors')


if OUTPUT_OVERLOAD:
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

model_dm = Doc2Vec(sentences=None, size=VEC_SIZE, window=WINDOW_SIZE, min_count=MIN_COUNT, workers=8, dm=1, sample=1e-4)
it = LTIterator('train_data_bow.csv', min_count=MIN_COUNT)
model_dm.build_vocab(it)
dm_total_wc = int(sum(v.count * v.sample_probability for k, v in iteritems(model_dm.vocab) if not k[0] in ['α', 'β', 'ζ']))
for _ in xrange(10):
    it = LTIterator('train_data_bow.csv', min_count=MIN_COUNT)
    model_dm.train(it, total_words=dm_total_wc)


model_dbow = Doc2Vec(sentences=None, size=VEC_SIZE, window=WINDOW_SIZE, min_count=MIN_COUNT, workers=8, dm=0, sample=1e-4)
it = LTIterator('train_data_bow.csv', min_count=MIN_COUNT)
model_dbow.build_vocab(it)
dbow_total_wc = int(sum(v.count * v.sample_probability for k, v in iteritems(model_dbow.vocab) if not k[0] in ['α', 'β', 'ζ']))
for __ in xrange(10):
    it = LTIterator('train_data_bow.csv', min_count=MIN_COUNT)
    model_dbow.train(it, total_words=dbow_total_wc)


# print('Building the real training vectors')
# Freeze the words
model_dm.train_words = False
model_dbow.train_words = False

# # Extend vocab with new labels
# it = LTIterator('train_data.csv', min_count=MIN_COUNT, only_return_starswith='α')
# dm_total_wc = add_labeled_texts(model_dm, it)
# it = LTIterator('train_data.csv', min_count=MIN_COUNT, only_return_starswith='α')
# dbow_total_wc = add_labeled_texts(model_dbow, it)


print('For labeled data extracting vectors for training the classifier')
# Extract the appropriate labels and concatenate their vectors
# Store the key and new vector in a list

VecTuple = namedtuple('VecTuple', ['label', 'vec'])

vec_tuples_train = []

for key, dm_itm in model_dm.vocab.items():
    if key.startswith('αþ') or key.startswith('αñ'):
        dm_idx = dm_itm.index
        dbow_idx = model_dbow.vocab[key].index
        vec_tuples_train.append(VecTuple(key, concatenate((model_dm.syn0[dm_idx], model_dbow.syn0[dbow_idx]))))


# Extract separated training data/targets from the list of labels and vectors
train_vectors = []
train_targets = []

for item in vec_tuples_train:
    if item[0].startswith('αþ'):
        train_targets.append(np_int(1))
        train_vectors.append(item[1])
    elif item[0].startswith('αñ'):
        train_targets.append(np_int(0))
        train_vectors.append(item[1])

train_vectors = array(train_vectors)

clf = SVC(C=50.0, gamma=.01, kernel='rbf')
clf.fit(train_vectors, train_targets)


print('Extending vocab and building vectors for new labels')

# Extend vocab with new labels
it = LTIterator('test_data_bow.csv', min_count=MIN_COUNT)
dm_total_wc = add_labeled_texts(model_dm, it)
it = LTIterator('test_data_bow.csv', min_count=MIN_COUNT)
dbow_total_wc = add_labeled_texts(model_dbow, it)

# Train the new labels
for ____ in xrange(1):
    it = LTIterator('test_data_bow.csv', min_count=MIN_COUNT)
    model_dm.train(it, total_words=dm_total_wc)
    it = LTIterator('test_data_bow.csv', min_count=MIN_COUNT)
    model_dbow.train(it, total_words=dbow_total_wc)


print('For test data extracting vectors for prediction')
# Extract the appropriate labels and concatenate their vectors
# Store the key and new vector in a list

vec_tuples_test = []

for key, dm_itm in model_dm.vocab.items():
    if key.startswith('βþ') or key.startswith('βñ'):
        dm_idx = dm_itm.index
        dbow_idx = model_dbow.vocab[key].index
        vec_tuples_test.append(VecTuple(key, concatenate((model_dm.syn0[dm_idx], model_dbow.syn0[dbow_idx]))))


# Extract separated training data/targets from the list of labels and vectors
test_vectors = []
test_targets = []

for item in vec_tuples_test:
    if item[0].startswith('βþ'):
        test_targets.append(np_int(1))
        test_vectors.append(item[1])
    elif item[0].startswith('βñ'):
        test_targets.append(np_int(0))
        test_vectors.append(item[1])

print('Now we predict sentiment of new labels')

# Grab prediction values
predicted = clf.predict(test_vectors)

# Send our best sentiments (bow)
acc = metrics.accuracy_score(test_targets, predicted)

print('Accuracy: ', str(acc * 100.0) + '%')


