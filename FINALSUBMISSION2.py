# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 16:59:34 2017

@author: DK
"""
from keras.utils.np_utils import to_categorical
import numpy as np
import pandas as pd
import re
import json
import itertools
from collections import Counter
import tensorflow as tf
import os
import time
import datetime
from tensorflow.contrib import learn
import csv
import multiprocessing as mp
import gensim.models
Word2Vec0 = gensim.models.KeyedVectors.load_word2vec_format('file://C:/Users/DK/Downloads/wiki.ru.vec', binary=False)#Word2Vec embedding
print('Word2Vec is loaded')
np.random.seed(0)
training_json = json.load(open("D://ted_ru-20160408train.json"))
testing_json=json.load(open("D://ted_ru-20160408test.json"))
total_tags=[]
total=[]
validation=[]
valid_id=[]
num_categories=8
def tagmaker(data, tags):
    for i in range(0,len(data)):
        words=data[i]['head']['keywords']
        if  'technology' in words and 'entertainment' in words and 'design' in words:
            tags.append(7)
        if  'technology' not in words and 'entertainment' in words and 'design' in words:
            tags.append(6)
        if  'technology' in words and 'entertainment' not in words and 'design' in words:
            tags.append(5)
        if  'technology' in words and 'entertainment' in words and 'design' not in words:
            tags.append(4)
        if  'technology' not in words and 'entertainment' not in words and 'design' in words:
            tags.append(3)
        if  'technology' not in words and 'entertainment' in words and 'design' not in words:
            tags.append(2)
        if  'technology' in words and 'entertainment' not in words and 'design' not in words:
            tags.append(1)
        if  'technology' not in words and 'entertainment' not in words and 'design' not in words:
            tags.append(0)
        print('Making tag for'+str(i))
def clean_str(string): #FUNCTION CLEANING TEXT
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя0-9(),!?\'\`]", " ", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\n", " \n ", string)
    return string.strip().lower()
def tokenize(data,toappend):#just removing punctuation
    for i in range(0,len(data)):
        toappend.append(clean_str(data[i]['content']))
#lemmatise and remove punctuation!!!!!!!!!!!!!!!!!1111111!!!
tagmaker(training_json,total_tags)
print('Tags are made')
tokenize(testing_json, validation)
tokenize(training_json,total)
print('Texts are properly cleaned')
for i in range(0,len(testing_json)):
    valid_id.append(testing_json[i]['@id'])
print('Validation ids for sending are appended')
total_tags=np.array(total_tags)
total=np.array(total)
validation=np.array(validation)
alpha=0.9#split train/test
testing=total[range(int(alpha*len(total)),len(total))]
training=total[range(0,int(alpha*len(total)))]
training_tags=total_tags[range(0,int(alpha*len(total)))]
testing_tags=total_tags[range(int(alpha*len(total)),len(total))]

print('Train&test sets are made')

from keras.layers import Convolution1D, MaxPooling1D, Flatten, Dense
from keras.models import Sequential
from nltk import word_tokenize
#import utils 
#from utils import ModelNotTrainedException
 
class CNNEmbeddedVecClassifier:
    def __init__(self,
                 classdict,
                 wvmodel=Word2Vec0,
                 n_gram=2,
                 vecsize=300,
                 nb_filters=1200,
                 maxlen=15):
        self.wvmodel = wvmodel
        self.classdict = classdict
        self.n_gram = n_gram
        self.vecsize = vecsize
        self.nb_filters = nb_filters
        self.maxlen = maxlen
        self.trained = False
 
    def convert_trainingdata_matrix(self):
        classlabels = self.classdict[0]
        texts=self.classdict[1]
 
        # tokenize the words, and determine the word length
        phrases = []
        indices = []
        #for label in classlabels:
        for ident in range(len(classlabels)):
            indices.append(to_categorical(classlabels[ident],num_categories))
            phrases.append(texts[ident].split())
        # store embedded vectors
        train_embedvec = np.zeros(shape=(len(phrases), self.maxlen, self.vecsize))
        for i in range(len(phrases)):
            for j in range(min(self.maxlen, len(phrases[i]))):
                train_embedvec[i, j] = self.word_to_embedvec(phrases[i][j])
        indices = np.array(indices, dtype=np.int)
        indices.shape=(indices.shape[0],num_categories)
        return classlabels, train_embedvec, indices
 
    def train(self):
        # convert classdict to training input vectors
        self.classlabels, train_embedvec, indices = self.convert_trainingdata_matrix()
        # build the deep neural network model
        model = Sequential()
        model.add(Convolution1D(filters=self.nb_filters,
                                kernel_size=self.n_gram,
                                padding='valid',
                                activation='relu',
                                input_shape=(self.maxlen, self.vecsize)))
        model.add(MaxPooling1D(pool_size=self.maxlen-self.n_gram+1))
        model.add(Flatten())
        model.add(Dense(num_categories, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
 
        # train the model
        model.fit(train_embedvec, indices)
 
        # flag switch
        self.model = model
        self.trained = True
 
    def word_to_embedvec(self, word):
        return self.wvmodel[word] if word in self.wvmodel else np.zeros(self.vecsize)
 
    def shorttext_to_matrix(self, shorttext):
        matrix = np.zeros((self.maxlen, self.vecsize))
        tokens=shorttext.split()
        for i in range(min(self.maxlen, len(tokens))):
            matrix[i] = self.word_to_embedvec(tokens[i])
        return matrix
 
    def predict(self, texts,return_probabilities=True):
        predictions=[]
        if not self.trained:
            raise ModelNotTrainedException()
 
        # retrieve vector
        for shorttext in texts:
            matrix = np.array([self.shorttext_to_matrix(shorttext)])  
            probabilities=self.model.predict(matrix) 
            # classification using the neural network
            if return_probabilities:
                predictions.append (probabilities )
            else:
                predictions.append(np.argmax(probabilities))
            
 
        return predictions


import argparse
import csv
import os
print('Making classifier for training set')
classdict = [(training_tags),(training)]#It is to be of
train_classifier = CNNEmbeddedVecClassifier(classdict,maxlen=3500)
train_classifier.train()
#train_classifier.savemodel("D:/TRAINCNN.txt")
#trainclassifier trained for 3500, trainclassifier1 trained for 2000 - all only on test data
Prediction=train_classifier.predict(testing,False)
acc=sum(Prediction==testing_tags)/len(Prediction)
print('Accuracy is '+str(acc))
print('Making final classifier')

classdict = [(total_tags),(total)]#It is to be of
classifier = CNNEmbeddedVecClassifier(classdict,maxlen=2000)
classifier.train()
#classifier.savemodel("D:/TOTALCNN.txt")
Prediction=classifier.predict(validation,False)

A1=pd.DataFrame({'id':valid_id, 'class':Prediction})   
A1[['id','class']].to_csv("D:/ans.csv",index=False)




