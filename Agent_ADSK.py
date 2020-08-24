#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 12:16:32 2019

@author: Kouppas
"""
# Importing the libraries
import pandas as pd
import numpy as np
import random
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.models import load_model
from keras.callbacks import EarlyStopping
from collections import deque

# Creating the policy to be followed by the agent
class Policy():
    def __init__(self, dimension, Neural_Net, done = False):
        
        # Setting Parameters
        self.done = done
        self.dimension = dimension
        self.discount = 0.5
        self.greedy = 1.0
        self.past_recollection = deque(maxlen = 2792)
        if self.done == False:
            self.classifier =  self.build_classifier()
        else:
            self.classifier = load_model(Neural_Net)

    # Creating the ANN that will predict the action taken by the agent
    def build_classifier(self):
        # Initialising the ANN 
        classifier = Sequential()
        # Adding the input layer and the first hidden layer
        classifier.add(Dense(units = 39, kernel_initializer = 'normal', activation = 'relu', input_dim = self.dimension))
        # Adding the second hidden layer
        classifier.add(Dense(units = 17, kernel_initializer = 'normal', activation = 'relu'))
        # Adding the third hidden layer
        classifier.add(Dense(units = 12, kernel_initializer = 'normal', activation = 'relu'))
        # Adding the output layer
        classifier.add(Dense(units = 3, kernel_initializer = 'normal', activation = 'linear'))
        # Compiling the ANN
        classifier.compile(optimizer = 'adam', loss = 'mean_squared_error')
        return classifier
#23*7*4*3 3*39+39*17+17*12+12*3

    # Take the next action to maximise reward
    def take_action(self, s):
        self.random_number = np.random.rand()
        if not self.done and self.random_number < self.greedy:
            return random.randrange(0, 3)
        selection = np.argmax(self.classifier.predict(s)[0])
        return selection
    
    # Creating a function that will help the agent learn from past experiences
    def learning_experience(self, batch):
        self.small_sample = random.sample(self.past_recollection, batch)
        for s, a, r, s_prime, done in self.small_sample:
            target_reward = r
            if not done:
                target_reward = r + self.discount * np.amax(self.classifier.predict(s_prime)[0])
            target_reward_false = self.classifier.predict(s)
            target_reward_false[0][a] = target_reward
            self.classifier.fit(s, target_reward_false, epochs = 50)#, callbacks = [EarlyStopping(monitor = 'loss', patience = 3)])
        if self.greedy > 0.01:
            self.greedy = self.greedy * 0.9965
    
#, verbose = 0

# dataset
# neural network maths
# make it start learning
