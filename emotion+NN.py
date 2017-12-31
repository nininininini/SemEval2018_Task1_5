
# coding: utf-8

# In[56]:


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

    return trainDF, devDF


class emotionNN:

    def __init__(self, trainDF, devDF, emotion, model):
        self.emotion = emotion
        self.model = model
        
        self.x_train = sequence.pad_sequences(np.array(trainDF['tweet_ids']), maxlen=MAX_LEN)
        #self.y_train = np.array(trainDF[self.emotion])
        self.y_train = np.array([trainDF['emotions']])[0]
        self.x_dev = sequence.pad_sequences(np.array(devDF['tweet_ids']), maxlen=MAX_LEN)
        #self.y_dev = np.array(devDF[self.emotion])
        self.y_dev = np.array([devDF['emotions']])[0]
        
        
        

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

        score, acc = self.model.evaluate(self.x_dev, self.y_dev)
        return score, acc

    def predict(self, testDF):
        x_test = sequence.pad_sequences(np.array(testDF['tweet_ids']), maxlen=MAX_LEN)
        predictions = self.model.predict(x_test)
        testDF['emotions' + '_pred'] = predictions
        tp = 0
        fp = 0
        tn = 0
        fn = 0

        for i, pred in enumerate(testDF['emotions' + '_pred']):
            if pred >= 0.5:
                if testDF['emotions'][i] == 1:
                    tp += 1
                else:
                    fp += 1
            if pred <= 0.5:
                if testDF['emotions'][i] == 1:
                    fn += 1
                else:
                    tn += 1
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = (precision * recall) / (precision + recall)

        print("F1: {}\nPrecision: {}\nRecall: {}".format(f1, precision, recall))



# In[57]:


data_dir = 'C:/Users/Oliver/PycharmProjects/SemEval2018_Task1_5/data/'
train_file = os.path.join(data_dir, '2018-E-c-En-train.txt')
dev_file = os.path.join(data_dir, '2018-E-c-En-dev.txt')

VOCAB_SIZE = 10000
MAX_LEN = 100
BATCH_SIZE = 32
EMBEDDING_SIZE = 20
HIDDEN_SIZE = 10
EPOCHS = 10 # Standard 10
UNKNOWN_TOKEN = "<unk>"


# In[58]:


trainDF, devDF = read_data(train_file, dev_file)


# In[59]:


#print(trainDF.iloc[:, -11:].values.tolist())


# In[60]:


trainDF['emotions'] = trainDF.iloc[:, -11:].values.tolist()
devDF['emotions'] = devDF.iloc[:, -11:].values.tolist()


# In[61]:


#print(trainDF[:2])


# In[62]:


from keras.layers import Conv1D, GlobalMaxPooling1D
cnn_model = Sequential()
cnn_model.add(Embedding(VOCAB_SIZE, EMBEDDING_SIZE))
cnn_model.add(Conv1D(2*HIDDEN_SIZE, kernel_size=3, activation='tanh', strides=1, padding='valid'))
cnn_model.add(GlobalMaxPooling1D())
cnn_model.add(Dense(HIDDEN_SIZE, activation='tanh'))
cnn_model.add(Dense(1, activation='sigmoid'))


# In[63]:


x_train = sequence.pad_sequences(np.array(trainDF['tweet_ids']), maxlen=MAX_LEN)
y_train = np.array([trainDF['emotions']])[0]
x_dev = sequence.pad_sequences(np.array(devDF['tweet_ids']), maxlen=MAX_LEN)
y_dev = np.array([devDF['emotions']])[0]


# In[64]:



x =np.array([trainDF['emotions']])
print(x[0].shape)


# In[65]:


print(x_train.shape)
print(x_train[:2])
print(y_train.shape)
print(y_train[:3])


# In[66]:


from keras.layers import Conv1D, GlobalMaxPooling1D
cnn_model = Sequential()
cnn_model.add(Embedding(VOCAB_SIZE, EMBEDDING_SIZE))
cnn_model.add(Conv1D(2*HIDDEN_SIZE, kernel_size=3, activation='tanh', strides=1, padding='valid'))
cnn_model.add(GlobalMaxPooling1D())
cnn_model.add(Dense(HIDDEN_SIZE, activation='tanh'))
cnn_model.add(Dense(y_train.shape[1], activation='sigmoid'))


# In[67]:


angerNN = emotionNN(trainDF, devDF, 'emotions', model)
score, acc = angerNN.run()
print("\nScore: {}, Accuracy: {}".format(score, acc))


# In[79]:


x_test = sequence.pad_sequences(np.array(devDF['tweet_ids']), maxlen=MAX_LEN)
predictions = cnn_model.predict(x_test)
print(predictions)




# In[90]:



tp = 0
fp = 0
tn = 0
fn = 0
all_correct = 0


# In[96]:


for i, pred in enumerate(predictions):
    for j, em in enumerate(pred):
        tmp = tp+ tn
        if em >= 0.5:
            if devDF['emotions'][i][j] == 1:
                tp += 1
            else:
                fp += 1
        if em <= 0.5:
            if devDF['emotions'][i][j] == 1:
                fn += 1
            else:
                tn += 1
        if tp + tn ==11:
            all_correct += 1
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = (precision * recall) / (precision + recall)

print("F1: {}\nPrecision: {}\nRecall: {}\nCompletely correct: {}".format(f1, precision, recall, all_correct))


# In[34]:


model = Sequential()
model.add(Dense(HIDDEN_SIZE, activation='relu', input_dim=x_train.shape[1]))
model.add(Dropout(0.1))
model.add(Dense(600, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(y_train.shape[1], activation='sigmoid'))



# In[35]:


sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy',
              optimizer=sgd)

model.fit(x_train, y_train, epochs=5, batch_size=2000)


# In[36]:


preds = model.predict(x_dev)
preds[preds>=0.5] = 1
preds[preds<0.5] = 0
print(preds)


# In[30]:




