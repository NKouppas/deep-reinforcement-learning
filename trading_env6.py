#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:30:44 2019

@author: Kouppas
"""
# Importing the libraries
import math
import gym
from gym import error, spaces, utils
import numpy as np
from gym.utils import seeding
import pandas as pd

# Creating the trading environment for the agent using OpenAI Gym
class TradingEnv6(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, trading_cost = 0.0025):
        # Setting the trading cost which is assumed to be 0.25 % of each transaction
        self.trading_cost = trading_cost
    
    # Creating the reset function which is used to reset the state of the agent
    def reset(self, data, window, t = 0):

        self.s = data[0: window]
        return self.s

    # Creating the step function which is used to determine the next state of the agent his reward the overall returns, and whether the episode has finished or not
    def step(self, data, df, t, window, a, j, returns, strategy_level, r, timesteps, Policy_done):
        self.long = 1
        self.short = 2
        self.flat = 0
        self.s_prime = data[t: window + t]   
        
        if a == self.long:
            returns.append(((df['PX_LAST'].iloc[j+window]/df['PX_LAST'].iloc[j+window-1]) - 1))
            strategy_level.append(strategy_level[j]*(1 + returns[j]))

        elif a == self.short:
            returns.append(-1*((df['PX_LAST'].iloc[j+window]/df['PX_LAST'].iloc[j+window-1]) - 1))
            strategy_level.append(strategy_level[j]*(1 + returns[j]))
                
        elif a == self.flat:
            returns.append(0*((df['PX_LAST'].iloc[j+window]/df['PX_LAST'].iloc[j+window-1]) - 1))
            strategy_level.append(strategy_level[j]*(1 + returns[j]))
            
        #if j == 0:  
            #r = ((strategy_level[j+1] / strategy_level[0])**(1/(j+1/365)) - 1)
        
       # else:
        r = strategy_level[j+1] - strategy_level[0] #((strategy_level[j+1] / strategy_level[0])**(1/(j+1/365)) - 1) #/ (np.std(returns)*(252)**0.5)    
                
        if j == timesteps - 1:
            done = True
            
        else:
            done = False
            
        return [self.s_prime, r, done]
    
    def render(self, mode='human', close=False):
        for i in range(3):
            print(i)
    


