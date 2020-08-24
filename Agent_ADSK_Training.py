#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:37:47 2019

@author: Kouppas
"""
import pandas as pd
import numpy as np
import Agent_ADSK
import matplotlib.pyplot as plt
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
sc = StandardScaler()
data = sc.fit_transform(df)
pca = PCA(n_components = 3)
data = pca.fit_transform(data)
#print(pca.explained_variance_ratio_)

# Defining parameters
oos = 2797
window = 2000#1044
features = 3
policy = Policy(features, "-")
timesteps = oos - window - 1
batch = 32
number_of_episodes = 2
Returns = []

for i in range(0, number_of_episodes):
    print("Episode " + str(i))
    s = environment.reset(data, window)
    returns = []
    strategy_level = [100]
#    r = 0
    
    for j in range(0, timesteps):
        a = policy.take_action(s)
        print(a)
        r = 0
        s_prime, r, done = environment.step(data, df, j + 1, window, a, j, returns, strategy_level, r, timesteps, policy.done)
        policy.past_recollection.append((s, a, r, s_prime, done))
        s = s_prime
        
    policy.learning_experience(batch)       
    
    #if i % 10 == 0:
    Returns.append(((strategy_level[796]/strategy_level[0])**(1/(796/365))-1)*100)
                
    #if i % 10 == 0:
    policy.classifier.save("ANN" + str(i))

# Episodes = []
# for i in range(0, number_of_episodes, 10):
#     Episodes.append(i)

# Plotting the overall returns over the number of episodes the agent was trained
# plt.plot(Episodes, Returns)
# plt.ylim(-51, 51)
# plt.xlim(0,500)
# plt.xlabel('Number of Episodes')
# plt.ylabel('Return Since Inception per Episode')
# plt.title('Convergence Graph')
# plt.show()

print(str(((strategy_level[796]/strategy_level[0])**(1/(796/365))-1)*100) + '%')


