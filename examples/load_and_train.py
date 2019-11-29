#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import sys

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parentdir)

import bilstm
import numpy as np

from bilstm.tagger import Tagger
from bilstm.models import BiLSTMCRF
from bilstm.preprocessing import IndexTransformer
from bilstm.utils import load_data_and_labels, save_params
from bilstm.trainer import Trainer
from keras import optimizers
from seqeval.metrics import f1_score
from sklearn.model_selection import train_test_split
from time import strftime, localtime


params = {'language': 'en',
          'use_valid': True,
          'optimizer': 'adam',
          'epochs': 1,
          'batch_size': 1,
          'save_model': True}

datasets = {'train_en': os.path.join(parentdir,
                                     'data/conll2003/en/ner/train.txt'),
            'valid_en': os.path.join(parentdir,
                                     'data/conll2003/en/ner/valid.txt'),
            'test_en': os.path.join(parentdir,
                                    'data/conll2003/en/ner/test.txt'),
            'train_tr': os.path.join(parentdir, 'data/tr/train_raw.txt'),
            'test_tr': os.path.join(parentdir, 'data/tr/test_raw.txt')}


saved_model = {'weights': "model_weights.h5",
               'params': "params.json",
               'preprocessor': "model_preprocessor.json",
               'config': "config.txt"}

loaded_model = {'weights': os.path.join(parentdir,
                                        "saved_models/model_weights.h5"),
                'params': os.path.join(parentdir, "saved_models/params.json"),
                'preprocessor': os.path.join(
                    parentdir, "saved_models/model_preprocessor.json"),
                'config': os.path.join(parentdir, "saved_models/config.txt")}


if __name__ == "__main__":
    print("Loading model and preprocessor...")

    model = BiLSTMCRF.load(loaded_model['weights'],
                           loaded_model['params'])
    it = IndexTransformer.load(loaded_model['preprocessor'])

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
    else:
        x_train, y_train = load_data_and_labels(datasets['train_en'])
        x_train, y_train = x_train[:200], y_train[:200]

        if params['use_valid']:
            x_valid, y_valid = load_data_and_labels(datasets['valid_en'])
        else:
            x_valid, y_valid = None, None

        x_test, y_test = load_data_and_labels(datasets['test_en'])

    model.compile(loss=model.get_loss(), optimizer=params['optimizer'])

    print("Training the model...")

    trainer = Trainer(model, preprocessor=it)
    trainer.train(x_train,
                  y_train,
                  x_valid=x_valid,
                  y_valid=y_valid,
                  epochs=params['epochs'],
                  batch_size=params['batch_size'])

    print("Tagging a sentence...")

    tagger = Tagger(model, preprocessor=it)

    if params['language'] in ('tr', 'TR', 'turkish', 'TURKISH'):
        text = "İstanbul'da, ağustos ayında yüzde doksan ihtimalle " +\
            "İstanbul Teknik Üniversitesi tarafından Teoman konseri " +\
            "saat beşte ücreti on lira olacak şekilde Ayazağa kampüsünde " +\
            "düzenlenecek."
    else:
        text = "EU rejects German call to boycott British celebrity " + \
            "John Locke to force Barcelona to play football in " + \
            "Championship League."

    res = tagger.analyze(text)
    print(res)

    print("Calculating accuracy...")

    x_test = it.transform(x_test)
    length = x_test[-1]
    y_pred = model.predict(x_test)
    y_pred = it.inverse_transform(y_pred, length)
    score = f1_score(y_test, y_pred)

    print("Test accuracy: ", score)

    if params['save_model']:
        print("Saving the model...")

        saving_time = strftime("%Y-%m-%d %H.%M.%S", localtime())
        directory = os.path.join(parentdir, 'saved_models/' + saving_time)

        if not os.path.exists(directory):
            os.makedirs(directory)

        weights_path = os.path.join(directory, saved_model['weights'])
        params_path = os.path.join(directory, saved_model['params'])
        preprocessor_path = os.path.join(directory, saved_model['preprocessor'])
        config_path = os.path.join(directory, saved_model['config'])

        model.save(weights_path,
                   params_path)
        it.save(preprocessor_path)

        params['score'] = score

        save_params(config_path, params)

    print("Finished!")
