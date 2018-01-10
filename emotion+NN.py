# coding: utf-8

# usr/bin/python3
"""
Todo:
understand multiclass neural network
11 classification networks?
outputs von em netzen als aux input für multiclass netz
emoji only als aux input für em netze

FRAGEN:
Wie Ergebnisse auswerten?
Ideas:
    Word normalization
    Emoji aux input
    train NN for each emotion
"""

import os
import pandas as pd
import numpy as np
from nltk.tokenize import TweetTokenizer
from collections import Counter

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Bidirectional, Dropout, Activation
from keras.optimizers import SGD
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras import regularizers, initializers
from time import time


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
    devDF = pd.read_csv(dev_file, sep='\t')

    allDF = pd.concat([trainDF, devDF], ignore_index=True)
    allDF = allDF.reindex(np.random.permutation(allDF.index))
    allDF.insert(1, 'tweet_tokenized', (allDF['Tweet'].apply(lambda x: tokenizer.tokenize(x))))

    word2id = create_dictionary(allDF["tweet_tokenized"], VOCAB_SIZE)

    allDF.insert(1, 'tweet_ids', (allDF['Tweet'].apply(lambda x: to_ids(x, dictionary=word2id))))

    allDF['all'] = allDF.iloc[:, -11:].values.tolist()
    total = len(allDF)
    trainend = int(total * 0.6)
    devend = trainend + int(total * 0.2)
    return allDF.iloc[:trainend, :], allDF.iloc[trainend:devend, :], allDF.iloc[devend:, :]


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
        print(predictions)
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        all_correct = 0
        labels = list(testDF['all'])
        for i, pred in enumerate(predictions):
            print(pred)
            print(labels[i])
            for j, em in enumerate(pred):
                if em >= 0.3:
                    if labels[i][j] == 1:
                        tp += 1
                    else:
                        fp += 1
                if em <= 0.3:
                    if labels[i][j] == 1:
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


start = time()
data_dir = 'data/'
train_file = os.path.join(data_dir, '2018-E-c-En-train.txt')
dev_file = os.path.join(data_dir, '2018-E-c-En-dev.txt')

VOCAB_SIZE = 100000
MAX_LEN = 100
BATCH_SIZE = 64
EMBEDDING_SIZE = 20
HIDDEN_SIZE = 10
EPOCHS = 10  # Standard 10
UNKNOWN_TOKEN = "<unk>"
EMOTIONS = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'love',
            'optimism', 'pessimism', 'sadness', 'surprise', 'trust']

trainDF, devDF, testDF = read_data(train_file, dev_file)
# print(len(trainDF) + len(devDF) + len(testDF))
# print("train__", len(trainDF))
# print(trainDF[:1]["ID"])
# print(trainDF[-1:]["ID"])
# print()
# print("dev__", len(devDF))
# print(devDF[:1]["ID"])
# print(devDF[-1:]["ID"])
# print()
# print("test__", len(testDF))
# print(testDF[:1]["ID"])
# print(testDF[-1:]["ID"])

#
# x_train = sequence.pad_sequences(np.array(trainDF['tweet_ids']), maxlen=MAX_LEN)
# x_dev = sequence.pad_sequences(np.array(devDF['tweet_ids']), maxlen=MAX_LEN)
#
# for emotion in EMOTIONS:
#     print("\nRunning CNN for emotion: {}".format(emotion))
#     y_train = np.array(trainDF[emotion])
#     y_dev = np.array(devDF[emotion])
#     eModel = create_cnn_model(emotion)
#     eNN = emotionNN(trainDF, devDF, eModel, emotion)
#     eNN.run()
#     predictions = eNN.model.predict(x_train)
#     trainDF[emotion+"_pred"] = predictions
#
# trainDF['all_pred'] = trainDF.iloc[:, -11:].values.tolist()
#
#



# model = create_cnn_model()

cnn_model = Sequential()
cnn_model.add(Embedding(VOCAB_SIZE, EMBEDDING_SIZE))
cnn_model.add(Conv1D(2 * HIDDEN_SIZE,
                     kernel_size=3,
                     activation='tanh',
                     strides=1,
                     padding='valid',
                     kernel_regularizer=regularizers.l1(0.001),))
cnn_model.add(GlobalMaxPooling1D())
cnn_model.add(Dense(HIDDEN_SIZE, activation='tanh'))
cnn_model.add(Dense(HIDDEN_SIZE, activation='tanh'))
cnn_model.add(Dense(HIDDEN_SIZE, activation='tanh'))
cnn_model.add(Dense(y_train.shape[1], activation='sigmoid'))


multiClassNN = emotionNN(trainDF, devDF, cnn_model)
score, acc = multiClassNN.run(verbose=2)
# multiClassNN.predict(testDF)
print("\nScore: {}, Accuracy: {}".format(score, acc))

multiClassNN.predict(testDF)
print("Runtime: {} seconds".format(time()-start))
