# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 22:23:32 2020

@author: CSY
"""


from Dino_A3C import A3C
import gym
import gym_chrome_dino
import numpy as np
import Image_Processing as ip
import time
import argparse



'''
To run the test, type:
"python test.py <saved actor weights h5 file name> <saved critic weights h5 file name>

Trained Demo will be presented during the presentation
'''

def __process_initial_state__(state, input_shape):
    canny = ip.img_processing("canny",input_shape,input_shape,state)
    stacked = ip.stack_images([canny,canny,canny,canny],4)
    reshaped_state = ip.reshape_to(4,stacked)
    return reshaped_state



def __process_new_state__(stacked_state, next_state, input_shape):
    canny = ip.img_processing("canny",input_shape,input_shape,next_state)
    new_stacked = ip.fifo_images(canny,stacked_state)
    return new_stacked



def __action__(model, state):
    policy = model.predict(state)[0]
    a = np.random.choice(2,1,p=policy)[0]
    return a



def __test__(a3c,env,input_shape):
    while True:
        done = False
        state = env.reset()
        state = __process_initial_state__(state, input_shape[0])
        while not done:
            env.render()
            action = __action__(a3c.actor_,state)
            print(action)
            next_state,reward,done,info = env.step(action)
            next_state = __process_new_state__(state,next_state,input_shape[0])
            state = next_state
            if done:
                break


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='load model.')
    parser.add_argument('actor',help='actor to test')
    parser.add_argument('critic',help='critic to test')
    args = parser.parse_args()
    actor = args.actor
    critic = args.critic


    environment = 'ChromeDino-v0'
    env = gym.make(environment)
    input_shape = (50,50,4)
    output_shape = env.action_space.n



    a3c = A3C(input_shape,output_shape,environment,1)
    a3c.actor_.load_weights(actor)
    a3c.critic_.load_weights(critic)
    
    
    __test__(a3c,env,input_shape)





































