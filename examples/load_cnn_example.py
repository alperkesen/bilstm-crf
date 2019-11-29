#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parentdir)

import bilstm
from bilstm.tagger import Tagger
from bilstm.models import BiLSTMCRF
from bilstm.preprocessing import IndexTransformer
from bilstm.utils import load_data_and_labels
from seqeval.metrics import f1_score


params = {'language': 'en'}

datasets = {'train_en': os.path.join(parentdir,
                                     'data/conll2003/en/ner/train.txt'),
            'valid_en': os.path.join(parentdir,
                                     'data/conll2003/en/ner/valid.txt'),
            'test_en': os.path.join(parentdir,
                                    'data/conll2003/en/ner/test.txt'),
            'train_tr': os.path.join(parentdir, 'data/tr/train_raw.txt'),
            'test_tr': os.path.join(parentdir, 'data/tr/test_raw.txt')}


loaded_model = {'weights': os.path.join(parentdir,
                                        "saved_models/model_weights.h5"),
                'params': os.path.join(parentdir, "saved_models/params.json"),
                'preprocessor': os.path.join(
                    parentdir, "saved_models/model_preprocessor.json")}


def main():
    print("Loading the model...")

    model = BiLSTMCRF.load(loaded_model['weights'],
                           loaded_model['params'])
    it = IndexTransformer.load(loaded_model['preprocessor'])

    tagger = Tagger(model, preprocessor=it)

    print("Tagging a sentence...")

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

    if params['language'] in ('tr', 'TR', 'turkish', 'TURKISH'):
        x_test, y_test = load_data_and_labels(datasets['test_tr'])
    else:
        x_test, y_test = load_data_and_labels(datasets['test_en'])

    x_test = it.transform(x_test)
    length = x_test[-1]
    y_pred = model.predict(x_test)
    y_pred = it.inverse_transform(y_pred, length)
    score = f1_score(y_test, y_pred)

    print("Test accuracy: ", score)


if __name__ == "__main__":
    main()
