# libraries for NLP
import nltk
# nltk.download('punkt')
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

# libraries for tensorflow
import tensorflow as tf
from tensorflow import keras
import numpy as np
import random
import json


# import chatbot intent files
with open('intents.json') as jsondata:
  intents = json.load(jsondata)


words = []
classes = []
documents = []
ignore = ['?'] # if we want to remove special character we can put here

for intent in intents['intents']:
  for ptrn in intent['patterns']:
    # tokenize every word in the sentence
    token = nltk.word_tokenize(ptrn)
    # print(token)
    # adding word to words list
    words.extend(token) # using append is not useful as it would have added the whole sentence as a list in the words, we want to push each word separatelya
    # adding words along with tag to documents list
    documents.append((token,intent['tag']))
    # adding tags to classes list
    if intent['tag'] not in classes:  # we want to append only unique tags
      classes.append(intent['tag'])

words = [stemmer.stem(w.lower()) for w in words if w not in ignore]
words = sorted(list(set(words))) # converting words to set so that repeating words gets removed and then again convert it into list from set then sort

classes = sorted(list(set(classes))) # convertint to set just to make sure that there are no repeating words

# creating training data
x = []
y = []

# creating an empty array for the output
outempty = [0] * len(classes) # [0] will be multiplied by # of classes available , [0] * 4 = [0 0 0 0]

# creating training set, bag of words for each sentence
for pair in documents:
  # instantiating bag of words
  bag = []

  # extracting sentence from pair containing (sentence,tag)
  pattern = pair[0]

  # converting the words in pattern to their base form
  pattern = [stemmer.stem(word.lower()) for word in pattern]

  # checking if words in pattern available in 'words(variable)', if yes then append 1 in bag else 0
  for w in words:
    bag.append(1) if w in pattern else bag.append(0)

  # changing outempty to list, it was already a list but just to make sure.
  outempty_updated = list(outempty)

  # assigning index places to 1 of outempty array wherever the tag is positioned in classes array
  outempty_updated[classes.index(pair[1])] = 1

  x.append([bag,outempty_updated])

random.shuffle(x)
x = np.array(x,dtype='object')
# xtrain = list(x[:,0])
# ytrain = list(x[:,1])
xtrain = list(x[:,0]) # contaiting words as x to train model
ytrain = list(x[:,1]) # containing classes as y to train model

xtrain = np.array(xtrain)
ytrain = np.array(ytrain)

model = keras.Sequential([
    keras.layers.Dense(10,input_shape = (len(xtrain[0]),)),
    keras.layers.Dense(80,activation = 'relu'),
    keras.layers.Dense(50,activation = 'relu'),
    keras.layers.Dense(20,activation = 'relu'),
    keras.layers.Dense(10,activation = 'relu'),
    keras.layers.Dense(len(ytrain[0]),activation = 'softmax')
])
model.compile(optimizer = 'adam',loss ='categorical_crossentropy',metrics = 'accuracy')
model.fit(xtrain,ytrain,epochs = 100)

model.save('chatbot.pkl')
