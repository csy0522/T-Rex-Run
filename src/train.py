# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 23:21:37 2020

@author: CSY
"""



from A3C_Dino import A3C

import gym
import gym_chrome_dino


if __name__ == "__main__":
    environment = 'ChromeDinoNoBrowser-v0'
    env = gym.make(environment)
    input_shape = (50,50,4)
    output_shape = env.action_space.n
    threads = 8
    
    
    env.close()
    
    a3c = A3C(input_shape,output_shape,threads,environment)
    a3c.__train__()