# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 23:21:37 2020

@author: CSY
"""



from A3C import A3C

import gym
import gym_chrome_dino


if __name__ == "__main__":
    env = gym.make('ChromeDinoNoBrowser-v0')
    
    input_shape = (50,50,4)
    output_shape = env.action_space.n
    gamma = 1
    threads = 8
    environment = env
    
    
    env.close()
    
    a3c = A3C(input_shape,output_shape,gamma,threads,environment)
    a3c.__train__()