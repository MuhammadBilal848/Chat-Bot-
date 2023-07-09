import tensorflow as tf
from tensorflow import keras
import pickle
import nltk
# nltk.download('punkt')
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
import numpy as np
import random
import json


chatbot = keras.models.load_model('chatbot.pkl')

data = pickle.load(open('wordnclass','rb'))
words = data['words']
classes = data['classes']

with open('intents.json') as jsondata:
  intents = json.load(jsondata)

def CleanUpSentence(sentence):
  ''' This function helps create a tokenizer for every word in the given sentence and convert the word to its base form. '''
  # tokenize the pattern
  sentence_tok = nltk.word_tokenize(sentence)

  # word to stem
  sentence_tok = [stemmer.stem(w.lower()) for w in sentence_tok ]
  return sentence_tok

def BagOfWords(sentence,words):
  ''' This function helps create a word vector for input by mapping the available words in a given sentence to pass 0*len(words). '''
  sentence_w = CleanUpSentence(sentence)

  # generating bag of words
  bag = [0] * len(words)  # words len is 52
  for word in sentence_w:
    for i,w in enumerate(words):
      if w == word: # checking that words in our sentence is available in 'words' that our model is trained on.
        # e.g first word 'where' from sentence is checked in words and if it is available then assign 1 on that place in bag array
        # similarly for 'are' 'you' 'locat' .
        bag[i] = 1

  bag = np.array(bag)
  return bag # The bag contains the accuracies of the given sentence being classified into one of the 8 classes.

def classify(sentence):
  ''' This function helps in classifying which class the sentence belongs to. '''
  # generate probabilities from model
  bag = BagOfWords(sentence,words) # The bag contains the accuracies of the given sentence being classified into one of the 8 classes.

  results = chatbot.predict(np.array([bag])) # model will predict based on word vector the class the words belong to

  results = [[i,r] for i,r in enumerate(results[0]) if r>0.3] # this has the index and accuracy of the highest class
  # print('before sort',results)
  results.sort(key = lambda x:x[1] , reverse = True) # sorting to make sure if there are more than 1 classes that are
  # greater than 0.3 so sort according to the second element of each array, so we are sorting based on 2nd element "x[1]" in each array of array
  # that is having accuracy.
  # if we had this : [[5, 0.81768185], [2, 0.31511548], [1, 0.41768185], [8, 0.51768185]]
  # the code will sort this into : [[5.0.81768185] , [8.0.51768185] , [1.0.41768185] , [2.0.31511548]]
  # print('after sort',results)
  return_list = []

  for r in results:
    return_list.append((classes[r[0]],r[1])) # appending class and accuracy

  return return_list

def response(sentence):
  results = classify(sentence)
  if results: # checking if result is empty(False) or not(True)
    while results:
      for i in intents['intents']:
        if i['tag'] == results[0][0]: # matching the class to tag and returning one of the random response in that class.
          return random.choice(i['responses'])

      results.pop(0)

print(response('Where are you located?'))

print(response('Hello'))

print(response('tell me about yourself?'))

print(response('can you me some movies?'))


# # pickle.dump({'words':words , 'classes':classes},open('wordnclass','wb'))

# chatbot = keras.models.load_model('chatbot.pkl')

# data = pickle.load(open('wordnclass','rb'))
# words = data['words']
# classes = data['classes']

# with open('intents.json') as jsondata:
#   intents = json.load(jsondata)

# def CleanUpSentence(sentence):
#   ''' This function helps create a tokenizer for every word in the given sentence and convert the word to its base form. '''
#   # tokenize the pattern
#   sentence_tok = nltk.word_tokenize(sentence)

#   # word to stem
#   sentence_tok = [stemmer.stem(w.lower()) for w in sentence_tok ]
#   return sentence_tok

# CleanUpSentence('Where are you located?')

# def BagOfWords(sentence,words):
#   ''' This function helps create a word vector for input by mapping the available words in a given sentence to pass 0*len(words). '''
#   sentence_w = CleanUpSentence(sentence)

#   # generating bag of words
#   bag = [0] * len(words)  # words len is 52
#   for word in sentence_w:
#     for i,w in enumerate(words):
#       if w == word: # checking that words in our sentence is available in 'words' that our model is trained on.
#         # e.g first word 'where' from sentence is checked in words and if it is available then assign 1 on that place in bag array
#         # similarly for 'are' 'you' 'locat' .
#         bag[i] = 1

#   bag = np.array(bag)
#   return bag # The bag contains the accuracies of the given sentence being classified into one of the 8 classes.

# b = BagOfWords('Where are you located',words)
# b

# res = chatbot.predict(np.array([b]))
# res[0]

# def classify(sentence):
#   ''' This function helps in classifying which class the sentence belongs to. '''
#   # generate probabilities from model
#   bag = BagOfWords(sentence,words) # The bag contains the accuracies of the given sentence being classified into one of the 8 classes.

#   results = chatbot.predict(np.array([bag])) # model will predict based on word vector the class the words belong to

#   results = [[i,r] for i,r in enumerate(results[0]) if r>0.3] # this has the index and accuracy of the highest class
#   # print('before sort',results)
#   results.sort(key = lambda x:x[1] , reverse = True) # sorting to make sure if there are more than 1 classes that are
#   # greater than 0.3 so sort according to the second element of each array, so we are sorting based on 2nd element "x[1]" in each array of array
#   # that is having accuracy.
#   # if we had this : [[5, 0.81768185], [2, 0.31511548], [1, 0.41768185], [8, 0.51768185]]
#   # the code will sort this into : [[5.0.81768185] , [8.0.51768185] , [1.0.41768185] , [2.0.31511548]]
#   # print('after sort',results)
#   return_list = []

#   for r in results:
#     return_list.append((classes[r[0]],r[1])) # appending class and accuracy

#   return return_list

# classify('where are you located?')

# for a in intents['intents']:
#   print(a)

# def response(sentence):
#   results = classify(sentence)
#   if results: # checking if result is empty(False) or not(True)
#     while results:
#       for i in intents['intents']:
#         if i['tag'] == results[0][0]: # matching the class to tag and returning one of the random response in that class.
#           return random.choice(i['responses'])

#       results.pop(0)

# response('Where are you located?')

# response('Hello')

# response('tell me about yourself?')

# response('can you me some movies?')
