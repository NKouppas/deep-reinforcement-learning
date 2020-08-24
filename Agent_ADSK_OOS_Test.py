#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:37:47 2019

@author: Kouppas
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Agent_ADSK
import gym
import Trading_Environment6
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
environment = gym.make('TradingEnv-v6')

# Importing the data
df = pd.read_csv('S&P500.csv', sep = ';')
df = df.drop('Dates', axis = 1)
df = df.dropna()
df = df[0:2797]
sc_oos = StandardScaler()
sc_oos.fit(df)
df = pd.read_csv('S&P500.csv', sep = ';')
df = df.drop('Dates', axis = 1)
df = df.dropna()
df = df[2797:]
data = sc_oos.transform(df)
pca = PCA(n_components = 3)
data = pca.fit_transform(data)
#print(pca.explained_variance_ratio_)

# Defining parameters
window = 2000#1044
features = 3
timesteps = len(data) - window - 1
number_of_episodes = 2

for i in range(1, number_of_episodes, 10):
    print("Episode number " + str(i))
    policy = Policy(features, "ANN" + str(i), True)
    s = environment.reset(data, window)
    returns = []
    strategy_level = [100]
        
    for j in range(0, timesteps):
        a = policy.take_action(s)
        print(a)
        r = 0
        s_prime, r, done = environment.step(data, df, j + 1, window, a, j, returns, strategy_level, r, timesteps, policy.done)
                
        policy.past_recollection.append((s, a, r, s_prime, done))
        s = s_prime
 
print(str(((strategy_level[249]/strategy_level[0])**(1/(249/365))-1)*100) + '%')


