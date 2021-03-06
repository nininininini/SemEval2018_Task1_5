# usr/bin/python3
"""
Todo:
Read in Data into Pandas Dataframe
create network for only 1 emotion based on previous tasks
K fold prediction?
understand multiclass neural network
11 classification networks?
nltk tweet tokenizer
"""

import os
import pandas as pd
import numpy as np
from nltk.tokenize import TweetTokenizer
from collections import Counter

import emoji

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Bidirectional
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

def extract_emojis(str):
  return [c for c in str if c in emoji.UNICODE_EMOJI]

def read_data(train_file, dev_file):
    tokenizer = TweetTokenizer()
    trainDF = pd.read_csv(train_file, sep='\t')
    trainDF = trainDF.reindex(np.random.permutation(trainDF.index))
    trainDF.insert(1, 'tweet_tokenized', (trainDF['Tweet'].apply(lambda x: tokenizer.tokenize(x))))
    trainDF.insert(1, 'emojis', (trainDF['tweet_tokenized'].apply(lambda x: extract_emojis(x))))

    devDF = pd.read_csv(dev_file, sep='\t')
    devDF = devDF.reindex(np.random.permutation(devDF.index))
    devDF.insert(1, 'tweet_tokenized', (devDF['Tweet'].apply(lambda x: tokenizer.tokenize(x))))
    devDF.insert(1, 'emojis', (devDF['tweet_tokenized'].apply(lambda x: extract_emojis(x))))

    word2id = create_dictionary(trainDF['tweet_tokenized'], VOCAB_SIZE)


    trainDF.insert(1, 'tweet_ids', (trainDF['Tweet'].apply(lambda x: to_ids(x, dictionary=word2id))))
    devDF.insert(1, 'tweet_ids', (devDF['Tweet'].apply(lambda x: to_ids(x, dictionary=word2id))))

    emoji2id = create_dictionary(trainDF['emojis'], 500)
    trainDF.insert(1, 'emoji_ids', (trainDF['emojis'].apply(lambda x: to_ids(x, dictionary=emoji2id))))
    devDF.insert(1, 'emoji_ids', (devDF['emojis'].apply(lambda x: to_ids(x, dictionary=emoji2id))))

    return trainDF, devDF


class emotionNN:

    def __init__(self, trainDF, devDF, emotion, model):
        self.emotion = emotion
        self.model = model

        self.x_train = sequence.pad_sequences(np.array(trainDF['tweet_ids']), maxlen=MAX_LEN)
        self.y_train = np.array(trainDF[self.emotion])
        self.x_dev = sequence.pad_sequences(np.array(devDF['tweet_ids']), maxlen=MAX_LEN)
        self.y_dev = np.array(devDF[self.emotion])

    def run(self):
        self.model.compile(loss='binary_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])
        self.model.fit(
            self.x_train,
            self.y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=(self.x_dev, self.y_dev),
            verbose=2
        )

        return self.model.evaluate(self.x_dev, self.y_dev)

    def predict(self, testDF):
        x_test = sequence.pad_sequences(np.array(testDF['tweet_ids']), maxlen=MAX_LEN)
        predictions = self.model.predict(x_test)
        testDF[self.emotion + '_pred'] = predictions
        tp = 0
        fp = 0
        tn = 0
        fn = 0

        for i, pred in enumerate(testDF[self.emotion + '_pred']):
            if pred >= 0.5:
                if testDF[self.emotion][i] == 1:
                    tp += 1
                else:
                    fp += 1
            if pred <= 0.5:
                if testDF[self.emotion][i] == 1:
                    fn += 1
                else:
                    tn += 1
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = (precision * recall) / (precision + recall)

        print("F1: {}\nPrecision: {}\nRecall: {}".format(f1, precision, recall))


if __name__ == "__main__":
    data_dir = 'C:/Users/Oliver/PycharmProjects/SemEval2018_Task1_5/data/'
    train_file = os.path.join(data_dir, '2018-E-c-En-train.txt')
    dev_file = os.path.join(data_dir, '2018-E-c-En-dev.txt')

    VOCAB_SIZE = 100000
    MAX_LEN = 100
    BATCH_SIZE = 32
    EMBEDDING_SIZE = 20
    HIDDEN_SIZE = 10
    EPOCHS = 10
    UNKNOWN_TOKEN = "<unk>"

    trainDF, devDF = read_data(train_file, dev_file)

    # lstm_model = Sequential()
    # lstm_model.add(Embedding(VOCAB_SIZE, EMBEDDING_SIZE))
    # lstm_model.add(Bidirectional(LSTM(HIDDEN_SIZE)))
    # lstm_model.add(Dense(2, activation='tanh'))
    # lstm_model.add(Dense(1, activation='sigmoid'))

    # angerNN = emotionNN(trainDF, devDF, 'anger', lstm_model)

    cnn_model = Sequential()
    cnn_model.add(Embedding(VOCAB_SIZE, EMBEDDING_SIZE))
    cnn_model.add(Conv1D(2 * HIDDEN_SIZE, kernel_size=3, activation='tanh', strides=1, padding='valid'))
    cnn_model.add(GlobalMaxPooling1D())
    cnn_model.add(Dense(HIDDEN_SIZE, activation='tanh'))
    cnn_model.add(Dense(1, activation='sigmoid'))

    angerNN = emotionNN(trainDF, devDF, 'anger', cnn_model)

    score, acc = angerNN.run()
    print("\nScore: {}, Accuracy: {}".format(score, acc))
    angerNN.predict(devDF)
