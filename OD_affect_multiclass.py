
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
from nltk.tokenize import TweetTokenizer
from collections import Counter

from keras.preprocessing import sequence
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Embedding, LSTM, Bidirectional, Dropout, Activation, Conv1D, GlobalMaxPooling1D, Input, concatenate
from keras import regularizers, initializers
from keras.optimizers import SGD, adam
from keras.callbacks import EarlyStopping, ModelCheckpoint



# In[2]:


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
    trainend = int(total * 0.8)
    devend = trainend + int(total * 0.1)
    return allDF.iloc[:trainend, :], allDF.iloc[trainend:devend, :], allDF.iloc[devend:, :]


def evaluate(predictions, y_test):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    all_correct = 0
    for i, pred in enumerate(predictions):
        for j, em in enumerate(pred):
            if em >= 0.5:
                if y_test[i][j] == 1:
                    tp += 1
                else:
                    fp += 1
            if em < 0.5:
                if y_test[i][j] == 1:
                    fn += 1
                else:
                    tn += 1
            if tp + tn == y_test.shape[1]:
                all_correct += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)

    print("F1: {}\nPrecision: {}\nRecall: {}\nCompletely correct: {}".format(f1, precision, recall, all_correct))


# Setzen der globalen Variablen
# 

# In[3]:


data_dir = 'data/'
train_file = os.path.join(data_dir, '2018-E-c-En-train.txt')
dev_file = os.path.join(data_dir, '2018-E-c-En-dev.txt')

VOCAB_SIZE = 100000  # Aufgrund der Kürze der Tweets: Große Vocab Size, damit der Anteil der unknown words nicht zu hoch ist
MAX_LEN = 100
BATCH_SIZE = 32
EMBEDDING_SIZE = 100
HIDDEN_SIZE = 50
EPOCHS = 100
UNKNOWN_TOKEN = "<unk>"
EMOTIONS = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'love',
            'optimism', 'pessimism', 'sadness', 'surprise', 'trust']


# Einlesen der Daten in Training-, Development-, und Testset
# 
# Die Daten werden in Pandas-Dataframes gespeichert, ursprünglich um auch die Daten der Emotionen einzeln verwenden zu können.
# 
# x_train, x_dev, x_test bestehen aus den Inputs als Word-ID-Vektoren.
# 
# y_train, y_dev, y_test bestehen aus den Output-Vektoren, mit jeweils 0 oder 1 für die einzelnen Emotionen.
# 
# 

# In[4]:


trainDF, devDF, testDF = read_data(train_file, dev_file)

x_train = sequence.pad_sequences(np.array(trainDF['tweet_ids']), maxlen=MAX_LEN)
x_dev = sequence.pad_sequences(np.array(devDF['tweet_ids']), maxlen=MAX_LEN)
x_test = sequence.pad_sequences(np.array(testDF['tweet_ids']), maxlen=MAX_LEN)

y_train = np.array([trainDF['all']])[0]
y_dev = np.array([devDF['all']])[0]
y_test = np.array([testDF['all']])[0]


# Aufbau des Neuronalen Netzes in der Keras Functional API
# 
# Getestet wurde auch ein bidirektionales LSTM, allerdings haben sich dort die Ergebnisse nicht verbessert und die Laufzeit stieg um das 30 bis 60-fache.
# 
# Um das Testen der Hyperparameter (zeitlich) zu erleichtern wurde das CNN weiter verwendet, auch weil es sehr gute Ergebnisse erzielte.
# 

# In[16]:


main_input = Input(shape=(MAX_LEN,), dtype='int32', name='main_input')

x = Embedding(VOCAB_SIZE, EMBEDDING_SIZE)(main_input)
x = Conv1D(
    filters=2 * HIDDEN_SIZE,
    kernel_size=3,
    activation='tanh',
    strides=1,
    padding='valid', )(x)
x = GlobalMaxPooling1D()(x)
x = Dropout(0.5)(x)

predictions = Dense(11, activation='sigmoid')(x)



# In[17]:


model = Model(inputs=main_input, outputs=predictions)


# Kompilieren des Modells
# 
# Hier wurden unterschiedliche Optimizer getestet (u.a. auch ein 'adam' Optimizer mit unterschiedlichen Lernraten).
# 
# Am besten performt hier allerdings rmsprop.

# In[18]:


model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'],
              )


# Implementierung von Early Stopping und Checkpoint nach Vorbild der in der Vorlesung präsentierten Folien.
# 

# In[19]:


early_stopper = EarlyStopping(monitor='val_acc', patience=3, mode='max')
checkpoint = ModelCheckpoint(data_dir + 'model.m', save_best_only=True, monitor='val_acc', mode='max')


# In[20]:


def compute_class_weights(y_train):
    class_totals = np.sum(y_train, axis=0)
    overall_sum = np.sum(y_train)
    class_weights = [1 + ((overall_sum - i) / overall_sum) ** 10 for i in class_totals]
    return class_weights


class_weights = compute_class_weights(y_train)
print(class_weights)


# In[21]:





# In[21]:


model.fit(
    x_train,
    y_train,
    batch_size=BATCH_SIZE,
    callbacks=[early_stopper, checkpoint],
    epochs=EPOCHS,
    validation_data=(x_dev, y_dev),
    class_weight=class_weights,
    verbose=1
)


# In[22]:


best_model = load_model(data_dir+'model.m')


# In[23]:


predictions = best_model.predict(x_test)


# Aufruf der Custom Evaluate Funktion, die F1 Score, Precision und Recall berechnet.
# 

# In[24]:


evaluate(predictions, y_test)



