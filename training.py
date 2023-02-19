import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

import keras
from keras.optimizers import SGD

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation, Dropout
# from tensorflow.python.keras.optimizers.experimental import SGD
# from tensorflow.keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

print(documents)

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))
# print(words)

classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
output_empty = [0] * 31

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    print(bag)
    output_row = list(output_empty)
    print(output_row)
    print(classes)
    print(classes.index(document[1]))
    output_row[classes.index(document[1])] = 1
    print(output_row)
    training.append([bag, output_row])
    print(training)

random.shuffle(training)
print(training)
training = np.array(training)


train_x = list(training[:, 0])
train_y = list(training[:, 1])

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(learning_rate=0.02, decay=0.000001, momentum=0.9, nesterov=True, name='SGD')
print(sgd)
model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])

hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('nelite_chatbot_model.h5', hist)

print('Done!')



