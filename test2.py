import collections
import random
import nltk
from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, LSTM, Bidirectional
import sys

# TODO
random.seed(111)


def nltk_data(n_texts_train=1500, n_texts_dev=500, vocab_size=10000):
    """
    Reads texts from the nltk movie_reviews corpus. A word2id dictionary is
    created and the words in the texts are substituted with their numbers. Training
    and Development data is returned, together with labels and the word2id dictionary.

    :param n_texts_train: the number of reviews that will form the training data
    :param n_texts_dev: the number of reviews that will form the development data
    :param vocab_size: the maximum size of the vocabulary.

    :return list texts_train: A list containing lists of wordids corresponding to
    training texts.
    :return list texts_dev: A list containing lists of wordids corresponding to
    development texts.
    :return labels_train: A list containing the labels (0 or 1) for the corresponding
    text entry in texts_train
    :return labels_dev: A ilst containing the labels (0 or 1) for the corresponding
    text entry in texts_dev
    :return word2id: The dictionary obtained from the training texts that maps each
    seen word to an id.
    """
    all_ids = movie_reviews.fileids()
    if (n_texts_train + n_texts_dev > len(all_ids)):
        print("Error: There are only", len(all_ids),
              "texts in the movie_reviews corpus. Training with all of those sentences.")
        n_texts_train = 1500
        n_texts_dev = 500
    posids = movie_reviews.fileids('pos')
    random.shuffle(all_ids)

    texts_train = []
    labels_train = []
    texts_dev = []
    labels_dev = []

    for i in range(n_texts_train):
        text = movie_reviews.raw(fileids=[all_ids[i]])
        tokens = [word.lower() for word in word_tokenize(text)]
        texts_train.append(tokens)
        if all_ids[i] in posids:
            labels_train.append(1)
        else:
            labels_train.append(0)

    for i in range(n_texts_train, n_texts_train + n_texts_dev):
        text = movie_reviews.raw(fileids=[all_ids[i]])
        tokens = [word.lower() for word in word_tokenize(text)]
        texts_dev.append(tokens)
        if all_ids[i] in posids:
            labels_dev.append(1)
        else:
            labels_dev.append(0)

    word2id = create_dictionary(texts_train, vocab_size)
    texts_train = [to_ids(s, word2id) for s in texts_train]
    texts_dev = [to_ids(s, word2id) for s in texts_dev]
    return (texts_train, labels_train, texts_dev, labels_dev, word2id)


def create_dictionary(texts, vocab_size):
    """
    Creates a dictionary that maps words to ids. More frequent words have lower ids.
    The dictionary contains at the vocab_size-1 most frequent words (and a placeholder '<unk>' for unknown words).
    The place holder has the id 0.
    """
    counter = collections.Counter()
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


VOCAB_SIZE = 10000
MAX_LEN = 100
BATCH_SIZE = 32
EMBEDDING_SIZE = 20
HIDDEN_SIZE = 10
EPOCHS = 10
UNKNOWN_TOKEN = "<unk>"

nltk.download('movie_reviews')
nltk.download('punkt')
x_train, y_train, x_dev, y_dev, word2id = nltk_data(vocab_size=VOCAB_SIZE)
x_train = sequence.pad_sequences(x_train, maxlen=MAX_LEN)
x_dev = sequence.pad_sequences(x_dev, maxlen=MAX_LEN)
print(x_train[:5])

# lstm_model = Sequential()
# lstm_model.add(Embedding(VOCAB_SIZE, EMBEDDING_SIZE))
# lstm_model.add(Bidirectional(LSTM(HIDDEN_SIZE)))
# lstm_model.add(Dense(2, activation='tanh'))
# lstm_model.add(Dense(1, activation='sigmoid'))
#
# lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# lstm_model.fit(
#     x_train,
#     y_train,
#     batch_size=BATCH_SIZE,
#     epochs=EPOCHS,
#     validation_data=(x_dev, y_dev)
# )
# score, acc = lstm_model.evaluate(x_dev, y_dev)
# print()
# print("LSTM Accuracy: ", acc)
#
#
# from keras.layers import Conv1D, GlobalMaxPooling1D
# cnn_model = Sequential()
# cnn_model.add(Embedding(VOCAB_SIZE, EMBEDDING_SIZE))
# cnn_model.add(Conv1D(2*HIDDEN_SIZE, kernel_size=3, activation='tanh', strides=1, padding='valid'))
# cnn_model.add(GlobalMaxPooling1D())
# cnn_model.add(Dense(HIDDEN_SIZE, activation='tanh'))
# cnn_model.add(Dense(1, activation='sigmoid'))
#
# cnn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# cnn_model.fit(
#     x_train,
#     y_train,
#     batch_size=BATCH_SIZE,
#     epochs=EPOCHS,
#     validation_data=(x_dev, y_dev)
# )
# score, acc = cnn_model.evaluate(x_dev, y_dev)
# print()
# print("CNN Accuracy: ", acc)
#
# from keras.layers import add, Input
#
# input = Input(shape=(MAX_LEN,))
# x = Embedding(VOCAB_SIZE, EMBEDDING_SIZE)(input)
#
# lstm_out = Bidirectional(LSTM(HIDDEN_SIZE))(x)
#
# cnn_out = Conv1D(2*HIDDEN_SIZE, kernel_size=3, activation='tanh', strides=1, padding='valid')(x)
# cnn_out = GlobalMaxPooling1D()(cnn_out)
#
# x = add([lstm_out, cnn_out])
# x = Dense(HIDDEN_SIZE, activation='tanh')(x)
# output = Dense(1, activation='sigmoid')(x)
#
# combined_model = Model(inputs=input, outputs=output)
# combined_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#
# combined_model.fit(
#     x_train,
#     y_train,
#     batch_size=BATCH_SIZE,
#     epochs=EPOCHS,
#     validation_data=(x_dev, y_dev)
# )
#
# score, acc = combined_model.evaluate(x_dev, y_dev)
# print()
# print("Composition Accuracy: ", acc)