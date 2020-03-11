# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 00:11:10 2020

@author: CSY
"""


import tensorflow as tf
from keras.layers import Input,Conv2D,BatchNormalization,Activation,Flatten,Dense
from keras.models import Model
from keras.optimizers import RMSprop
from keras import backend as K

import threading
import cv2 as cv
import gym
import gym_chrome_dino
from collections import deque
import numpy as np
import matplotlib.pyplot as plt

import Image_Processing as ip



EPISODE = 0
MAX_EPISODE = 100
TOTAL_REWARDS = deque()




class A3C:
    
    def __init__(self,input_shape,output_shape,threads,environment):
        self.input_shape_ = input_shape
        self.output_shape_ = output_shape
        self.gamma_ = 0.995
        self.threads_ = threads
        self.environment_ = environment
        
        self.actor_,self.critic_ = self.__build_actor_critic__()
        
        self.sess_ = tf.InteractiveSession()
        K.set_session(self.sess_)
        self.sess_.run(tf.global_variables_initializer())
        
        
        
    def __build_actor_critic__(self):
        input_ = Input(shape=self.input_shape_)
        conv = Conv2D(filters=32,kernel_size=(8,8),padding="valid",strides=(2,2))(input_)
        conv = BatchNormalization()(conv)
        conv = Activation('relu')(conv)
        conv = Conv2D(filters=32,kernel_size=(8,8),padding="valid",strides=(2,2))(conv)
        conv = BatchNormalization()(conv)
        conv = Activation('relu')(conv)
        conv = Conv2D(filters=32,kernel_size=(8,8),padding="valid",strides=(2,2))(conv)
        conv = BatchNormalization()(conv)
        conv = Activation('relu')(conv)
        
        flat = Flatten()(conv)
        
        dense = Dense(units=512)(flat)
        dense = Activation('relu')(dense)
        dense = Dense(units=128)(dense)
        dense = Activation('relu')(dense)
        
        policy = Dense(units=self.output_shape_, activation='softmax')(dense)
        value = Dense(units=1,activation='linear')(dense)
        
        actor = Model(inputs=input_,outputs=policy)
        critic = Model(inputs=input_,outputs=value)
        
        actor._make_predict_function()
        critic._make_predict_function()
        
        return actor,critic
    
    
    
    def __update_actor__(self):
        action = K.placeholder(shape=(None, self.output_shape_))
        advantages = K.placeholder(shape=(None, ))
        policy = self.actor_.output
        
        good_prob = K.sum(action * policy, axis=1)
        eligibility = K.log(good_prob+1e-10) * K.stop_gradient(advantages)
        loss = -K.sum(eligibility)
        
        gradient = RMSprop()
        updates = gradient.get_updates(self.actor_.trainable_weights,[],loss)
        train = K.function([self.actor_.input,action,advantages],[self.actor_.output],updates=updates)
        
        return train
    
    
    
    def __update_critic__(self):
        discounted_rewards = K.placeholder(shape=(None, ))
        value = self.critic_.output
        
        loss = K.mean(K.square(discounted_rewards - value))
        
        gradient = RMSprop()
        updates = gradient.get_updates(self.critic_.trainable_weights, [], loss)
        train = K.function([self.critic_.input, discounted_rewards],[self.critic_.output],updates=updates)
        
        return train
    
    
    
    def __train__(self):
        dinos = [Dino(self.input_shape_, self.output_shape_, self.environment_, self.actor_, self.critic_,
                     self.__update_actor__(), self.__update_critic__(), self.gamma_, self.sess_) 
                 for i in range(self.threads_)]
        for dino in dinos:
            dino.start()
            # dino.run()

    
    
    
    
    
class Dino(threading.Thread):
# class Dino:
    
    def __init__(self,input_shape,output_shape,environment,global_actor,global_critic,
                update_actor,update_critic,gamma,sess):
        threading.Thread.__init__(self)
        
        self.input_shape_ = input_shape
        self.output_shape_ = output_shape
        self.environment_ = environment
        self.global_actor_ = global_actor
        self.global_critic_ = global_critic
        self.update_actor_ = update_actor
        self.update_critic_ = update_critic
        self.gamma_ = gamma
        self.sess_ = sess

        
        self.states_, self.actions_, self.rewards_ = deque(),deque(),deque()
        
        
        
    def run(self):
        self.__train__()

        
        
    def __train__(self):
        global EPISODE
        env = gym.make(self.environment_)
        while EPISODE != MAX_EPISODE:
            done = False
            total_reward = 0
            state = env.reset()
            state = self.__process_initial_state__(state)
            while not done:
                action = self.__action__(state)
                next_state,reward,done,info = env.step(action)
                self.__remember__(state,action,reward)
                total_reward += reward
                next_state = self.__process_new_state__(state,next_state)
                state = next_state
                if done:
                    EPISODE += 1
                    print("episode: {}/{}\n reward: {}".format(EPISODE,MAX_EPISODE,total_reward))
                    TOTAL_REWARDS.append(total_reward)
                    self.__update_actor_critic__(total_reward > 600)
                    break
        self.__plot_total_rewards__()
        self.__save_weights__("{}".format(MAX_EPISODE))
    
    
    
    
    def __process_initial_state__(self, state):
        canny = ip.img_processing("canny",self.input_shape_[0],self.input_shape_[0],state)
        stacked = ip.stack_images([canny,canny,canny,canny],4)
        reshaped_state = ip.reshape_to(4,stacked)
        return reshaped_state
    
    
    
    def __process_new_state__(self, stacked_state, next_state):
        canny = ip.img_processing("canny",self.input_shape_[0],self.input_shape_[0],next_state)
        new_stacked = ip.fifo_images(canny,stacked_state)
        return new_stacked
    
    
    
    def __action__(self,state):
        policy = self.global_actor_.predict(state)[0]
        a = np.random.choice(self.output_shape_,1,p=policy)[0]
        return a
    
    
    
    def __discounted_rewards__(self, rewards, done):
        discounted_rewards = np.zeros_like(rewards)
        R = self.global_critic_.predict(self.states_[-1]) * (1-int(done))
        
        for i in reversed(range(len(rewards))):
            R = rewards[i] + self.gamma_ * R
            discounted_rewards[i] = R
        
        return discounted_rewards
    
    
    
    def __update_actor_critic__(self, done):
        states = ip.reshape_to(4,np.array(self.states_))
        # actions = np.array(self.actions_)
        value = ip.reshape_to(1,self.global_critic_.predict(states))
        discounted_rewards = self.__discounted_rewards__(self.rewards_, done)
        advantages = discounted_rewards - value
        
        self.update_actor_([states,self.actions_,advantages])
        self.update_critic_([states,discounted_rewards])
        self.__clear_deque__()
        
        
        
    def __clear_deque__(self):
        self.states_.clear()
        self.actions_.clear()
        self.rewards_.clear()
        
        
        
    def __remember__(self,state,action,reward):
        self.states_.append(state)
        act = np.zeros(self.output_shape_)
        act[action] = 1
        self.actions_.append(act)
        self.rewards_.append(reward)
        
        
        
    def __save_weights__(self,name):
        self.global_actor_.save_weights(name+"_actor.h5")
        self.global_critic_.save_weights(name+"_critic.h5")
        
        
        
    def __plot_total_rewards__(self):
        plt.figure(figsize=(10,8))
        plt.title("Reward / Episode")
        plt.xlabel("epochs")
        plt.ylabel("rewards")
        plt.plot(TOTAL_REWARDS, label="rewards")
        plt.legend()
        plt.savefig("plot.png")



    
