# -*- coding: utf-8 -*-
from collections import namedtuple
import pickle
from math import sqrt
import heapq

# From package scikit-learn
from sklearn.svm import SVC
from sklearn.metrics import metrics
from sklearn.linear_model import SGDClassifier

from numpy import concatenate, zeros
from utls.doc2vec import Doc2Vec
import ujson
from numpy import array, int as np_int, uint32, uint8, random, empty, float32 as REAL
from six import iteritems, itervalues
from six.moves import xrange
from utls.word2vec import Vocab

# see paper "Distributed Representations of Sentences and Documents"
#   http://cs.stanford.edu/~quocle/paragraph_vector.pdf
# Authors: Quoc Le, Tomas Mikolov

print('An attempt to reproduce something like "Experiment 3.2: IMDB sentiment", from the paper.\n'
      'This is gonna take a while, probably faster ways to build a reasonable classifier.')

LabeledText = namedtuple('LabeledText', ['text', 'labels'])


# Generator that can pre-pad the sentences with nulls, like in the paper.
class LTIterator(object):
    def __init__(self, fname, min_count):
        self.fname = fname
        self.min_count = min_count
        self.fobj = open(self.fname)
        # self.break_out_at = 10000
        # self.loop_count = 0

    def __iter__(self):
        for line in self.fobj:
            # self.loop_count += 1
            # if self.loop_count > self.break_out_at:
            #     break
            l_text, l_label = ujson.loads(line[:-1] if line.endswith('\n') else line)
            sentence_length = len(l_text)
            if self.min_count and sentence_length < self.min_count:
                l_text = (['null'] * (self.min_count - sentence_length)) + l_text
            yield LabeledText(l_text, l_label)


def add_labels(doc2vec_obj, sentences):
    """
    Extends vocabulary labels from a sequence of sentences (can be a once-only generator stream).
    Each sentence must be a LabeledText-like object
    We don't want a new vocab, so we need something different from whats packaged w/ word2vec
    """
    orig_len_vocab = len(doc2vec_obj.vocab)
    orig_total_words = sum(v.count for v in itervalues(doc2vec_obj.vocab))
    threshold_count = float(doc2vec_obj.sample) * orig_total_words
    sentence_no, vocab = -1, {}
    for sentence_no, sentence in enumerate(sentences):
        sentence_length = len(sentence.text)
        for label in sentence.labels:
            if label in vocab:
                vocab[label].count += sentence_length
            else:
                vocab[label] = Vocab(count=sentence_length)
    # assign a unique index to each new vocab item
    for word, v in iteritems(vocab):
        if v.count >= doc2vec_obj.min_count:
            v.index = len(doc2vec_obj.vocab)
            prob = (sqrt(v.count / threshold_count) + 1) * (threshold_count / v.count) if doc2vec_obj.sample else 1.0
            v.sample_probability = prob
            doc2vec_obj.index2word.append(word)
            doc2vec_obj.vocab[word] = v
    # add new vocab items to your hs tree
    if doc2vec_obj.hs:
        heap = list(itervalues(vocab))
        heapq.heapify(heap)
        for i in xrange(orig_len_vocab, len(doc2vec_obj.vocab) - 1):
            min1, min2 = heapq.heappop(heap), heapq.heappop(heap)
            heapq.heappush(heap, Vocab(count=min1.count + min2.count, index=i + len(doc2vec_obj.vocab), left=min1, right=min2))
        if heap:
            max_depth, stack = 0, [(heap[0], [], [])]
            while stack:
                node, codes, points = stack.pop()
                if node.index < len(doc2vec_obj.vocab):
                    # leaf node => store its path from the root
                    node.code, node.point = codes, points
                    max_depth = max(len(codes), max_depth)
                else:
                    # inner node => continue recursion
                    points = array(list(points) + [node.index - len(doc2vec_obj.vocab)], dtype=uint32)
                    stack.append((node.left, array(list(codes) + [0], dtype=uint8), points))
                    stack.append((node.right, array(list(codes) + [1], dtype=uint8), points))
    else:
        raise BaseException('PC Load Letter')
    # extend the model's vector sets
    random.seed(doc2vec_obj.seed)
    doc2vec_obj.syn0 = concatenate((doc2vec_obj.syn0, empty((len(doc2vec_obj.vocab) - orig_len_vocab, doc2vec_obj.layer1_size), dtype=REAL)))
    for i in xrange(orig_len_vocab, len(doc2vec_obj.vocab)):
        doc2vec_obj.syn0[i] = (random.rand(doc2vec_obj.layer1_size) - 0.5) / doc2vec_obj.layer1_size
    if doc2vec_obj.hs:
        doc2vec_obj.syn1 = concatenate((doc2vec_obj.syn1, zeros((len(doc2vec_obj.vocab) - orig_len_vocab, doc2vec_obj.layer1_size), dtype=REAL)))
    doc2vec_obj.syn0norm = None


VEC_SIZE = 400
WINDOW_SIZE = 10
MIN_COUNT = 9

print('Building the 75k training vectors')

model_dm = Doc2Vec(sentences=None, size=VEC_SIZE, window=WINDOW_SIZE, min_count=MIN_COUNT, workers=8, dm=1, sample=1e-4)
it = LTIterator('train_data.csv', min_count=MIN_COUNT)
model_dm.build_vocab(it)
it = LTIterator('train_data.csv', min_count=MIN_COUNT)
model_dm.train(it)

model_dbow = Doc2Vec(sentences=None, size=VEC_SIZE, window=WINDOW_SIZE, min_count=MIN_COUNT, workers=8, dm=0, sample=1e-4)
it = LTIterator('train_data.csv', min_count=MIN_COUNT)
model_dbow.build_vocab(it)
it = LTIterator('train_data.csv', min_count=MIN_COUNT)
model_dbow.train(it)

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

print('Now building a classifier for our initial test, how does it do on pre-computed vectors.')
# The paper uses a neural network, whatever that is...

clf = SVC(C=50.0, kernel='linear')

# For our first test we use a subset of train data
clf.fit(train_vectors[:20000], train_targets[:20000])

print('Without loading in new stuff, lets get an idea of what we can do.')
predicted = clf.predict(train_vectors[20000:25000])
acc = metrics.accuracy_score(train_targets[20000:25000], predicted)
print('Accuracy: ', str(acc * 100.0) + '%')

print('Now we got some new reviews coming in.\n'
      'But before we read then lets rebuild the classifier with all available data.')

del clf
clf = SVC(C=50.0, kernel='linear')
clf.fit(train_vectors, train_targets)


print('Extending vocab and building vectors for new labels')

# Freeze the words,should only matter for dm (high inflection)?
model_dm.train_words = False
model_dbow.train_words = False

# Extend vocab with new labels
it = LTIterator('test_data.csv', min_count=MIN_COUNT)
add_labels(model_dm, it)
it = LTIterator('test_data.csv', min_count=MIN_COUNT)
add_labels(model_dbow, it)

# Train the new labels
it = LTIterator('test_data.csv', min_count=MIN_COUNT)
model_dm.train(it)
it = LTIterator('test_data.csv', min_count=MIN_COUNT)
model_dbow.train(it)

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

# Send your best sentiments (bow)
acc = metrics.accuracy_score(test_targets, predicted)

print('Accuracy: ', str(acc * 100.0) + '%')