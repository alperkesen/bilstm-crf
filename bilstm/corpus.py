#!/usr/bin/env python
# -*- coding: utf-8 -*-


import sys
from gensim.models import Word2Vec


CORPUS_NAME = '../data/temp/corpus.txt'
EMBEDDING_DIM = 100
MIN_COUNT = 1
WINDOW_SIZE = 5
NUM_WORKERS = 4
ALGORITHM = 1  # CBOW (0), Skip-gram (1)


class Corpus(object):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        for line in open(self.filename, 'r', encoding='utf-8',
                         errors='ignore'):
            yield line.rsplit()


def write_file(model, filename='output.txt'):
    with open(filename, 'w', encoding='utf-8') as output_file:
        for word in model.wv.vocab.keys():
            vector = " ".join([str(x) for x in list(model.wv[word])])
            output_file.write(word + ' ' + vector + '\n')


if __name__ == '__main__':
    corpus = Corpus(CORPUS_NAME)
    model = Word2Vec(corpus,
                     size=EMBEDDING_DIM,
                     min_count=MIN_COUNT,
                     window=WINDOW_SIZE,
                     workers=NUM_WORKERS,
                     sg=ALGORITHM)
    
    output_filename = "output.txt"

    if len(sys.argv) > 1:
        output_filename = sys.argv[1]

    write_file(model, output_filename)
