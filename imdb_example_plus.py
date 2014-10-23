# -*- coding: utf-8 -*-
from collections import namedtuple
import logging
from gensim.models.doc2vec import Doc2Vec
from scipy.sparse import csr_matrix, hstack, vstack
import scipy.sparse.compressed
from sklearn.metrics import metrics
from sklearn.svm import SVC
from utls.imdb_example_utils import add_labeled_texts, train_model, LTIterator
from six.moves import xrange
from utls.text_svm import BOWObject

from numpy import array, int as np_int, concatenate, asarray


print('The Objective here is to combine Doc2Vec dense-vectors and BOW sparse vectors to produce\n'
      '\tbetter accuracy in determining IMDB sentiment')

# see paper "Distributed Representations of Sentences and Documents"
#   http://cs.stanford.edu/~quocle/paragraph_vector.pdf
# Authors: Quoc Le, Tomas Mikolov

VEC_SIZE = 40
WINDOW_SIZE = 10
MIN_COUNT = 9
NUM_ITERS = 5
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
train_model(model_dm, 'train_data.csv', num_iters=NUM_ITERS, min_count=MIN_COUNT)


model_dbow = Doc2Vec(sentences=None, size=VEC_SIZE, window=WINDOW_SIZE, min_count=MIN_COUNT, workers=8, dm=0, sample=1e-4)
train_model(model_dbow, 'train_data.csv', num_iters=NUM_ITERS, min_count=MIN_COUNT)

# Freeze the words
model_dm.train_words = False
model_dbow.train_words = False

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


# Now that we have our dense data, we build sparse data to append to it

print('Building train tdidf vectors')
bow_data_obj = BOWObject('train_data_bow.csv')
bow_data_obj.build_train()

# Extract separated training data/targets from the list of labels and vectors
train_dvectors = []
train_svectors = []
train_targets = []

cur_idx = 0

print('Preparing train data')
for item in vec_tuples_train:
    if item[0].startswith('αþ'):
        train_targets.append(np_int(1))
        train_dvectors.append(item[1])
        train_svectors.append(bow_data_obj.train[item[0]][1])
    elif item[0].startswith('αñ'):
        train_targets.append(np_int(0))
        train_dvectors.append(item[1])
        train_svectors.append(bow_data_obj.train[item[0]][1])

# We are going to want to keep these sparse matrices compressed by using sparse matrix tools
train_vectors = hstack([asarray(train_dvectors), vstack(train_svectors)], format='csr')
train_vectors.asfptype()

print('Training classifier')
# Train a classifier
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
    it = LTIterator('test_data.csv', min_count=MIN_COUNT)
    model_dm.train(it, total_words=dm_total_wc)
    it = LTIterator('test_data.csv', min_count=MIN_COUNT)
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


print('Building test tdidf vectors')
bow_data_obj.build_test('test_data_bow.csv')

# Extract separated training data/targets from the list of labels and vectors
test_dvectors = []
test_svectors = []
test_targets = []

print('Preparing test data')
for item in vec_tuples_test:
    if item[0].startswith('βþ'):
        test_targets.append(np_int(1))
        test_dvectors.append(item[1])
        test_svectors.append(bow_data_obj.test[item[0]][1])
    elif item[0].startswith('βñ'):
        test_targets.append(np_int(0))
        test_dvectors.append(item[1])
        test_svectors.append(bow_data_obj.test[item[0]][1])

test_vectors = hstack([asarray(test_dvectors), vstack(test_svectors)], format='csr')
test_vectors.asfptype()

print('Now we predict sentiment of new labels')

# Grab prediction values
predicted = clf.predict(test_vectors)

# Send our best sentiments (bow)
acc = metrics.accuracy_score(test_targets, predicted)

print('Accuracy: ', str(acc * 100.0) + '%')