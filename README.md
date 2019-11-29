# BiLSTM-CRF model for Turkish NER

**bilstm** is a Python library for named entity recognition, implemented in Keras. 

This library is modified version of [Anago](https://github.com/Hironsan/anago).


## Installation

You can install bilstm-crf from the repository:

```bash
$ git clone git@bitbucket.org/bilstm-crf.git
$ cd bilstm-crf
```

## Get Started

First, create a virtual environment for the project in the and install the required packages.

```bash
$ virtualenv .
$ source ./bin/activate
$ pip install -r requirements.txt
```

In the examples folder, you can run cnn_example.py to train the model. 

In order to choose the language of the data set, whether to use cnn and 
the hyperparameters of the model, you can change the values in the cnn_example file.

```
$ python examples/cnn_example.py
```

In bilstm-crf, the simplest type of model is the `Sequence` model. 
Sequence model includes essential methods like `fit`, `score`, `analyze` and `save`/`load`.
For more complex features, you should use the bilstm-crf modules such as `models`, `preprocessing` and so on.

Here is the data loader:

```python
>>> from bilstm.utils import load_data_and_labels

>>> x_train, y_train = load_data_and_labels('train.txt')
>>> x_test, y_test = load_data_and_labels('test.txt')
>>> x_train[0]
['EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'lamb', '.']
>>> y_train[0]
['B-ORG', 'O', 'B-MISC', 'O', 'O', 'O', 'B-MISC', 'O', 'O']
```

You can now iterate on your training data in batches:

```python
>>> import bilstm

>>> model = bilstm.Sequence()
>>> model.fit(x_train, y_train, epochs=15)
Epoch 1/15
541/541 [==============================] - 166s 307ms/step - loss: 12.9774
...
```

Evaluate your performance in one line:

```python
>>> model.score(x_test, y_test)
80.20  # f1-micro score
# For more performance, you have to use pre-trained word embeddings.
```

Or tagging text on new data:

```python
>>> text = 'President Obama is speaking at the White House.'
>>> model.analyze(text)
{
    "words": [
        "President",
        "Obama",
        "is",
        "speaking",
        "at",
        "the",
        "White",
        "House."
    ],
    "entities": [
        {
            "beginOffset": 1,
            "endOffset": 2,
            "score": 1,
            "text": "Obama",
            "type": "PER"
        },
        {
            "beginOffset": 6,
            "endOffset": 8,
            "score": 1,
            "text": "White House.",
            "type": "LOC"
        }
    ]
}
```

To download a pre-trained model, call `download` function:

```python
>>> from bilstm.utils import download

>>> url = 'https://storage.googleapis.com/chakki/datasets/public/ner/model_en.zip'
>>> download(url)
'Downloading...'
'Complete!'
>>> model = bilstm.Sequence.load('weights.h5', 'params.json', 'preprocessor.pickle')
>>> model.score(x_test, y_test)
90.61
```

## Feature Support

bilstm-crf supports following features:

* Model Training
* Model Evaluation
* Tagging Text
* Custom Model Support
* Downloading pre-trained model
* GPU Support
* Character feature
* CRF Support
* Custom Callback Support

bilstm-crf officially supports Python 3.4–3.6.


## Data and Word Vectors

Training data takes a tsv format.
The following text is an example of training data for English:

```
EU	B-ORG
rejects	O
German	B-MISC
call	O
to	O
boycott	O
British	B-MISC
lamb	O
.	O

Peter	B-PER
Blackburn	I-PER
```

Examples for Turkish training data is given in the below:

```
Ayvalık	LOCATION
,	O
Türkiye'nin	LOCATION
büyük	O
patronlarının	O
yöreye	O
duyduğu	O
ilgiden	O
memnun	O
```

bilstm-crf supports pre-trained word embeddings like [GloVe vectors](https://nlp.stanford.edu/projects/glove/) 
and [pretrained Turkish vectors](http://vectors.nlpl.eu/repository/).

## Reference

This library uses both bidirectional LSTM + CRF model based on
[Neural Architectures for Named Entity Recognition](https://arxiv.org/abs/1603.01360) by Lample, Guillaume, et al., NAACL 2016 
and LSTM + CNN + CRF model based on [End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF](https://arxiv.org/abs/1603.01354)
by Ma and Hovy, ACL 2016.
