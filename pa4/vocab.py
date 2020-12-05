# Taken from https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/image_captioning/build_vocab.py
import copy
import csv
import json
import os
import pickle
from collections import Counter

import nltk
from pycocotools.coco import COCO

from .utils import tenumerate

# A simple wrapper class for Vocabulary. No changes are required in this file
class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word.lower() in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word.lower()]

    def __len__(self):
        return len(self.word2idx)


def load_vocab(json, threshold):
    if os.path.isfile('saved_vocab.pkl'):
        with open('saved_vocab.pkl', 'rb') as f:
            vocab, loaded_threshold = pickle.load(f)
            if loaded_threshold == threshold:
                return vocab
            print("Using the saved vocab.")

    vocab = build_vocab(json, threshold)
    with open('saved_vocab.pkl', 'wb') as f:
        pickle.dump((vocab, threshold), f)
        print("Saved the vocab.")

    return vocab


def build_vocab(json, threshold):
    coco = COCO(json)
    counter = Counter()
    ids = coco.anns.keys()
    for i, id_ in tenumerate(ids):
        caption = str(coco.anns[id_]['caption'])
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)

        if tenumerate == enumerate and (i + 1) % 1000 == 0:
            print("[{}/{}] Tokenizing Captions.".format(i + 1, len(ids)))

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Add the words to the vocabulary.
    for word in words:
        vocab.add_word(word)
    return vocab
