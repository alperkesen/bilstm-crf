#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

from keras import optimizers
from pprint import pprint
from sklearn.model_selection import train_test_split
from time import strftime, localtime

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parentdir)

import bilstm
from bilstm.utils import load_data_and_labels, load_glove, load_tr_vectors, \
    load_tr_vectors2, save_params
from bilstm.corpus import Corpus


params = {'language': 'en',
          'use_pretrained': False,
          'use_valid': True,
          'use_cnn': False,
          'use_crf': True,
          'use_test_callback': False,
          'use_extra_layer': False,
          'learn_mode': 'marginal',
          'shuffle': True,
          'save_model': True,
          'valid_split': 0.2,
          'word_embedding_dim': 100,
          'char_embedding_dim': 25,
          'word_lstm_size': 100,
          'char_lstm_size': 25,
          'fc_dim': 100,
          'dropout': 0.5,
          'filter_size': 3,
          'filter_length': 30,
          'use_char': True,
          'initial_vocab': None,
          'optimizer': 'adam',
          'epochs': 15,
          'batch_size': 32,
          'verbose': 1}

datasets = {'train_en': os.path.join(parentdir,
                                     'data/conll2003/en/ner/train.txt'),
            'valid_en': os.path.join(parentdir,
                                     'data/conll2003/en/ner/valid.txt'),
            'test_en': os.path.join(parentdir,
                                    'data/conll2003/en/ner/test.txt'),
            'train_tr': os.path.join(parentdir, 'data/tr/train_raw.txt'),
            'test_tr': os.path.join(parentdir, 'data/tr/test_raw.txt')}

embeddings_paths = {'glove_en': os.path.join(
    parentdir, 'data/glove.6B/glove.6B.100d.txt'),
                    'tr_vec': os.path.join(parentdir, 'data/tr/tr.vec'),
                    'nlpl_vec': os.path.join(parentdir, 'data/tr/nlpl.txt')}


saved_model = {'weights': "model_weights.h5",
               'params': "params.json",
               'preprocessor': "model_preprocessor.json",
               'config': "config.txt"}


def main():
    print("Loading datasets...")

    if params['language'] in ('tr', 'TR', 'turkish', 'TURKISH'):
        X, y = load_data_and_labels(datasets['train_tr'])

        if params['use_valid']:
            x_train, x_valid, y_train, y_valid = train_test_split(
                X, y, test_size=params['valid_split'], random_state=42)
        else:
            x_train, y_train = X, y
            x_valid, y_valid = None, None

        x_test, y_test = load_data_and_labels(datasets['test_tr'])

        if params['use_pretrained']:
            corpus = Corpus(embeddings_paths['nlpl_vec'])
            embeddings = load_tr_vectors2(corpus)
            params['word_embedding_dim'] = 100
        else:
            embeddings = None
    else:
        x_train, y_train = load_data_and_labels(datasets['train_en'])

        if params['use_valid']:
            x_valid, y_valid = load_data_and_labels(datasets['valid_en'])
        else:
            x_valid, y_valid = None, None

        x_test, y_test = load_data_and_labels(datasets['test_en'])

        if params['use_pretrained']:
            embeddings = load_glove(embeddings_paths['glove_en'])
            params['word_embedding_dim'] = 100
        else:
            embeddings = None

    print("Building model...")

    model = bilstm.Sequence(word_embedding_dim=params['word_embedding_dim'],
                            char_embedding_dim=params['char_embedding_dim'],
                            word_lstm_size=params['word_lstm_size'],
                            char_lstm_size=params['char_lstm_size'],
                            fc_dim=params['fc_dim'],
                            dropout=params['dropout'],
                            embeddings=embeddings,
                            filter_size=params['filter_size'],
                            filter_length=params['filter_length'],
                            use_char=params['use_char'],
                            use_crf=params['use_crf'],
                            use_cnn=params['use_cnn'],
                            use_extra_layer=params['use_extra_layer'],
                            learn_mode=params['learn_mode'],
                            initial_vocab=params['initial_vocab'],
                            optimizer=params['optimizer'])

    print('Training the model...')

    if not params['use_test_callback']:
        x_callback, y_callback = None, None
    else:
        x_callback, y_callback = x_test, y_test

    model.fit(x_train, y_train,
              x_valid=x_valid, y_valid=y_valid,
              x_test=x_callback, y_test=y_callback,
              epochs=params['epochs'],
              batch_size=params['batch_size'],
              verbose=params['verbose'])

    print('Calculating accuracy...')

    model_score = model.score(x_test, y_test)

    print("Test accuracy: ", model_score)

    if params['save_model']:
        print('Saving the model...')

        saving_time = strftime("%Y-%m-%d %H.%M.%S", localtime())
        directory = os.path.join(parentdir, 'saved_models/' + saving_time)

        if not os.path.exists(directory):
            os.makedirs(directory)

        weights_path = os.path.join(directory, saved_model['weights'])
        params_path = os.path.join(directory, saved_model['params'])
        preprocessor_path = os.path.join(directory, saved_model['preprocessor'])
        config_path = os.path.join(directory, saved_model['config'])

        model.save(weights_path,
                   params_path,
                   preprocessor_path)

        params['score'] = model_score

        save_params(config_path, params)


if __name__ == '__main__':
    pprint(params)
    main()
