# coding: utf-8

# usr/bin/python3
"""
Todo:
understand multiclass neural network
11 classification networks?
outputs von em netzen als aux input für multiclass netz
emoji only als aux input für em netze
"""

import os
# import keras
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from timeit import timeit
from nltk.tokenize import TweetTokenizer
from collections import Counter

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Bidirectional, Dropout, Activation
from keras.optimizers import SGD
from keras.layers import Conv1D, GlobalMaxPooling1D


def create_dictionary(texts, vocab_size):
    """
    Creates a dictionary that maps words to ids. More frequent words have lower ids.
    The dictionary contains at the vocab_size-1 most frequent words (and a placeholder '<unk>' for unknown words).
    The place holder has the id 0.
    """
    counter = Counter()
    for tokens in texts:
        counter.update(tokens)
    vocab = [w for w, c in counter.most_common(vocab_size - 1)]
    word_to_id = {w: (i + 1) for i, w in enumerate(vocab)}
    word_to_id[UNKNOWN_TOKEN] = 0
    return word_to_id


def to_ids(words, dictionary):
    """
    Takes a list of words and converts them to ids using the word2id dictionary.
    """
    ids = []
    for word in words:
        ids.append(dictionary.get(word, dictionary[UNKNOWN_TOKEN]))
    return ids


def read_data(train_file, dev_file):
    tokenizer = TweetTokenizer()
    trainDF = pd.read_csv(train_file, sep='\t')
    trainDF = trainDF.reindex(np.random.permutation(trainDF.index))
    trainDF.insert(1, 'tweet_tokenized', (trainDF['Tweet'].apply(lambda x: tokenizer.tokenize(x))))

    devDF = pd.read_csv(dev_file, sep='\t')
    devDF = devDF.reindex(np.random.permutation(devDF.index))
    devDF.insert(1, 'tweet_tokenized', (devDF['Tweet'].apply(lambda x: tokenizer.tokenize(x))))

    word2id = create_dictionary(trainDF["tweet_tokenized"], VOCAB_SIZE)

    trainDF.insert(1, 'tweet_ids', (trainDF['Tweet'].apply(lambda x: to_ids(x, dictionary=word2id))))
    devDF.insert(1, 'tweet_ids', (devDF['Tweet'].apply(lambda x: to_ids(x, dictionary=word2id))))

    trainDF['all'] = trainDF.iloc[:, -11:].values.tolist()
    devDF['all'] = devDF.iloc[:, -11:].values.tolist()

    return trainDF, devDF


class emotionNN:

    def __init__(self, trainDF, devDF, model, emotion='all'):
        self.emotion = emotion
        self.model = model

        self.x_train = sequence.pad_sequences(np.array(trainDF['tweet_ids']), maxlen=MAX_LEN)

        self.x_dev = sequence.pad_sequences(np.array(devDF['tweet_ids']), maxlen=MAX_LEN)

        if self.emotion == 'all':
            self.y_train = np.array([trainDF['all']])[0]
            self.y_dev = np.array([devDF['all']])[0]
        else:
            self.y_train = np.array(trainDF[self.emotion])
            self.y_dev = np.array(devDF[self.emotion])

    def run(self, verbose=0):
        self.model.compile(loss='binary_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])
        self.model.fit(
            self.x_train,
            self.y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=(self.x_dev, self.y_dev),
            verbose=verbose
        )

        score, acc = self.model.evaluate(self.x_dev, self.y_dev)
        return score, acc

    def predict(self, testDF):
        x_test = sequence.pad_sequences(np.array(testDF['tweet_ids']), maxlen=MAX_LEN)
        predictions = self.model.predict(x_test)
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        all_correct = 0
        for i, pred in enumerate(predictions):
            for j, em in enumerate(pred):
                tmp = tp + tn
                if em >= 0.5:
                    if testDF['all'][i][j] == 1:
                        tp += 1
                    else:
                        fp += 1
                if em <= 0.5:
                    if testDF['all'][i][j] == 1:
                        fn += 1
                    else:
                        tn += 1
                if tp + tn == 11:
                    all_correct += 1
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = (precision * recall) / (precision + recall)

        print("F1: {}\nPrecision: {}\nRecall: {}\nCompletely correct: {}".format(f1, precision, recall, all_correct))


def create_cnn_model(emotion='all'):
    cnn_model = Sequential()
    cnn_model.add(Embedding(VOCAB_SIZE, EMBEDDING_SIZE))
    cnn_model.add(Conv1D(2 * HIDDEN_SIZE, kernel_size=3, activation='relu', strides=1, padding='valid'))
    cnn_model.add(GlobalMaxPooling1D())
    cnn_model.add(Dense(HIDDEN_SIZE, activation='relu'))
    if emotion == 'all':
        cnn_model.add(Dense(y_train.shape[1], activation='sigmoid'))
    else:
        cnn_model.add(Dense(1, activation='sigmoid'))
    return cnn_model


data_dir = 'C:/Users/Oliver/PycharmProjects/SemEval2018_Task1_5/data/'
train_file = os.path.join(data_dir, '2018-E-c-En-train.txt')
dev_file = os.path.join(data_dir, '2018-E-c-En-dev.txt')

VOCAB_SIZE = 10000
MAX_LEN = 100
BATCH_SIZE = 64
EMBEDDING_SIZE = 20
HIDDEN_SIZE = 10
EPOCHS = 5  # Standard 10
UNKNOWN_TOKEN = "<unk>"
EMOTIONS = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'love',
            'optimism', 'pessimism', 'sadness', 'surprise', 'trust']

trainDF, devDF = read_data(train_file, dev_file)

x_train = sequence.pad_sequences(np.array(trainDF['tweet_ids']), maxlen=MAX_LEN)
x_dev = sequence.pad_sequences(np.array(devDF['tweet_ids']), maxlen=MAX_LEN)

for emotion in EMOTIONS:
    print("Running CNN for emotion: {}".format(emotion))
    y_train = np.array(trainDF[emotion])
    y_dev = np.array(devDF[emotion])
    eModel = create_cnn_model(emotion)
    eNN = emotionNN(trainDF, devDF, eModel, emotion)
    eNN.run()
    predictions = eNN.model.predict(x_train)
    trainDF[emotion+"_pred"] = predictions

trainDF['all_pred'] = trainDF.iloc[:, -11:].values.tolist()



y_train = np.array([trainDF['all']])[0]
y_dev = np.array([devDF['all']])[0]

model = create_cnn_model()
multiClassNN = emotionNN(trainDF, devDF, model)
score, acc = multiClassNN.run(verbose=2)
print("\nScore: {}, Accuracy: {}".format(score, acc))

multiClassNN.predict(trainDF)
