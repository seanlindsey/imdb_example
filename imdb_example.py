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
import logging

# see paper "Distributed Representations of Sentences and Documents"
#   http://cs.stanford.edu/~quocle/paragraph_vector.pdf
# Authors: Quoc Le, Tomas Mikolov

print('An attempt to reproduce something like "Experiment 3.2: IMDB sentiment", from the paper.\n'
      '\tThis is gonna take a while, probably faster ways to build a reasonable classifier.')

LabeledText = namedtuple('LabeledText', ['text', 'labels'])


# Generator that can pre-pad the sentences with nulls, like in the paper.
class LTIterator(object):
    def __init__(self, fname, min_count):
        self.fname = fname
        self.min_count = min_count
        self.fobj = open(self.fname)
        self.extend_labels = False
        # self.break_out_at = 1000
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
            if self.extend_labels and l_label and l_label[0][0] == 'α':
                if l_label[0].startswith('αþ'):
                    l_label.append('ζpos')
                elif l_label[0].startswith('αñ'):
                    l_label.append('ζneg')
            yield LabeledText(l_text, l_label)


# TODO: test, add comments
def get_labeled_text(doc2vec_obj, sentence):
    if doc2vec_obj.hs:
        orig_len_vocab = len(doc2vec_obj.vocab)
        sentence_length = len(sentence.text)
        threshold_count = float(doc2vec_obj.sample) * len(doc2vec_obj.vocab)
        added_label_items = []
        for label in sentence.labels:
            if label not in doc2vec_obj.vocab:
                label_v = Vocab(count=sentence_length)
                label_v.index = len(doc2vec_obj.vocab)
                label_v.sample_probability = (sqrt(label_v.count / threshold_count) + 1) * (threshold_count / label_v.count) if doc2vec_obj.sample else 1.0
                doc2vec_obj.index2word.append(label_v)
                doc2vec_obj.vocab[label] = label_v
                added_label_items.append(label_v)
        if added_label_items:
            doc2vec_obj.create_binary_tree()
            # heap = list(itervalues(doc2vec_obj.vocab))
            # heapq.heapify(heap)
            # for i in xrange(len(doc2vec_obj.vocab) - 1):
            #     min1 = heapq.heappop(heap)
            #     min2 = heapq.heappop(heap)
            #     heapq.heappush(heap, Vocab(count=min1.count + min2.count, index=i + len(doc2vec_obj.vocab), left=min1, right=min2))
            #     if heap:
            #         max_depth, stack = 0, [(heap[0], [], [])]
            #         while stack:
            #             node, codes, points = stack.pop()
            #             if node.index < len(doc2vec_obj.vocab):
            #                 # leaf node => store its path from the root
            #                 node.code, node.point = codes, points
            #                 max_depth = max(len(codes), max_depth)
            #             else:
            #                 # inner node => continue recursion
            #                 points = array(list(points) + [node.index - len(doc2vec_obj.vocab)], dtype=uint32)
            #                 stack.append((node.left, array(list(codes) + [0], dtype=uint8), points))
            #                 stack.append((node.right, array(list(codes) + [1], dtype=uint8), points))
            random.seed(doc2vec_obj.seed)
            doc2vec_obj.syn0 = concatenate((doc2vec_obj.syn0, empty((len(doc2vec_obj.vocab) - orig_len_vocab, doc2vec_obj.layer1_size), dtype=REAL)))
            for i in xrange(orig_len_vocab, len(doc2vec_obj.vocab)):
                doc2vec_obj.syn0[i] = (random.rand(doc2vec_obj.layer1_size) - 0.5) / doc2vec_obj.layer1_size
            if doc2vec_obj.hs:
                doc2vec_obj.syn1 = concatenate((doc2vec_obj.syn1, zeros((len(doc2vec_obj.vocab) - orig_len_vocab, doc2vec_obj.layer1_size), dtype=REAL)))
            doc2vec_obj.syn0norm = None
            doc2vec_obj.train([sentence], total_words=1)
        rv = {label: doc2vec_obj.syn0[doc2vec_obj.vocab[label].index] for label in sentence.labels}
    else:
        raise BaseException('SET SOFT MODEL')
    # TODO: might be nice to clean up labels
    return rv


def add_labeled_texts(doc2vec_obj, sentences):
    """
    Extends vocabulary labels from a sequence of sentences (can be a once-only generator stream).
    Each sentence must be a LabeledText-like object
    We don't want a new vocab, so we need something different from whats packaged w/ word2vec
    """
    orig_len_vocab = len(doc2vec_obj.vocab)
    orig_total_words = sum(v.count for v in itervalues(doc2vec_obj.vocab))
    threshold_count = float(doc2vec_obj.sample) * orig_total_words
    sentence_no, vocab = -1, {}
    rv_word_count = 1
    for sentence_no, sentence in enumerate(sentences):
        sentence_length = len(sentence.text)
        rv_word_count += int(rv_word_count * ((sqrt(rv_word_count / threshold_count) + 1) * (threshold_count / rv_word_count) if doc2vec_obj.sample else 1.0))
        for label in sentence.labels:
            if label not in doc2vec_obj.vocab:
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
    # add new vocab items to our hs tree
    if doc2vec_obj.hs:
        # doc2vec_obj.create_binary_tree()
        heap = list(itervalues(doc2vec_obj.vocab))
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
        raise BaseException('PC LOAD LETTER')
    # extend the model's vector sets
    random.seed(doc2vec_obj.seed)
    doc2vec_obj.syn0 = concatenate((doc2vec_obj.syn0, empty((len(doc2vec_obj.vocab) - orig_len_vocab, doc2vec_obj.layer1_size), dtype=REAL)))
    for i in xrange(orig_len_vocab, len(doc2vec_obj.vocab)):
        doc2vec_obj.syn0[i] = (random.rand(doc2vec_obj.layer1_size) - 0.5) / doc2vec_obj.layer1_size
    if doc2vec_obj.hs:
        doc2vec_obj.syn1 = concatenate((doc2vec_obj.syn1, zeros((len(doc2vec_obj.vocab) - orig_len_vocab, doc2vec_obj.layer1_size), dtype=REAL)))
    doc2vec_obj.syn0norm = None
    return rv_word_count


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
it = LTIterator('train_data.csv', min_count=MIN_COUNT)
model_dm.build_vocab(it)
dm_total_wc = int(sum(v.count * v.sample_probability for k, v in iteritems(model_dm.vocab) if not k[0] in ['α', 'β', 'ζ']))
for _ in xrange(5):
    it = LTIterator('train_data.csv', min_count=MIN_COUNT)
    model_dm.train(it, total_words=dm_total_wc)


model_dbow = Doc2Vec(sentences=None, size=VEC_SIZE, window=WINDOW_SIZE, min_count=MIN_COUNT, workers=8, dm=0, sample=1e-4)
it = LTIterator('train_data.csv', min_count=MIN_COUNT)
model_dbow.build_vocab(it)
dbow_total_wc = int(sum(v.count * v.sample_probability for k, v in iteritems(model_dbow.vocab) if not k[0] in ['α', 'β', 'ζ']))
for __ in xrange(5):
    it = LTIterator('train_data.csv', min_count=MIN_COUNT)
    model_dbow.train(it, total_words=dbow_total_wc)


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

if TRY_WITH_PREBUILD:
    print('Now building a classifier for our initial test, how does it do on pre-computed vectors.')
    # The paper uses a neural network, whatever that is...

    clf = SVC(C=50.0, kernel='linear')

    # For our first test we use a subset of train data
    clf.fit(train_vectors[:20000], train_targets[:20000])

    print('Without loading in new stuff, lets get an idea of what we can do.')
    predicted = clf.predict(train_vectors[20000:25000])
    acc = metrics.accuracy_score(train_targets[20000:25000], predicted)
    print('Accuracy: ', str(acc * 100.0) + '%')
    del clf
    print('Now we got some new reviews coming in.\n'
          '\tBut before we read them lets rebuild the classifier with all available data.')
else:
    print('Now building a classifier')

clf = SVC(C=50.0, kernel='linear')
clf.fit(train_vectors, train_targets)


print('Extending vocab and building vectors for new labels')

# Freeze the words,should only matter for dm (high inflection)?
model_dm.train_words = False
model_dbow.train_words = False

# Extend vocab with new labels
it = LTIterator('test_data.csv', min_count=MIN_COUNT)
dm_total_wc = add_labeled_texts(model_dm, it)
it = LTIterator('test_data.csv', min_count=MIN_COUNT)
dbow_total_wc = add_labeled_texts(model_dbow, it)

# Train the new labels
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

if TEST_GET_VEC:
    print('Try playing with adding stuff one by one')

    # Make the LabeledText
    good_rev = LabeledText(['good', 'wonderful', 'enlightening', 'great', 'movie', ',', 'powerful', 'performance', ',', 'i', 'plan', 'on', 'watching', 'it', 'again', '.'], ['βþ100010110110'])
    bad_rev = LabeledText(['bad', 'poor', 'terrible', 'awful', 'movie', ',', 'never', 'watching', 'it', 'again', '.'], ['βñ100010110110'])

    # Push the vectors into our models
    good_rev_labels_dm = get_labeled_text(model_dm, good_rev)
    good_rev_labels_dbow = get_labeled_text(model_dbow, good_rev)
    bad_rev_labels_dm = get_labeled_text(model_dm, bad_rev)
    bad_rev_labels_dbow = get_labeled_text(model_dbow, bad_rev)

    # Grab the vector indexes from the models
    good_rev_dm_idx = model_dm.vocab['βþ100010110110'].index
    good_rev_dbow_idx = model_dbow.vocab['βþ100010110110'].index
    bad_rev_dm_idx = model_dm.vocab['βñ100010110110'].index
    bad_rev_dbow_idx = model_dbow.vocab['βñ100010110110'].index

    # Prepare the vectors for our classifier
    good_rev_vec = concatenate((model_dm.syn0[good_rev_dm_idx], model_dbow.syn0[good_rev_dbow_idx]))
    bad_rev_vec = concatenate((model_dm.syn0[bad_rev_dm_idx], model_dbow.syn0[bad_rev_dbow_idx]))

    # Print out what we get from classifier
    print('We should see, [1 0], for good and bad')
    print(clf.predict([good_rev_vec, bad_rev_vec]))

