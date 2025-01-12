import retro
from gym import Env 
from gym.spaces import MultiBinary, Box, MultiDiscrete
import numpy as np
import cv2
import math


# Custom environment wrapper to preprocess and reward-shape
class StreetFighter(Env): 
    def __init__(self):
        super().__init__()
        
        self.observation_space = Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)  #(84, 84, 1)
        #self.action_space = MultiDiscrete(np.array([2] * 12))
        self.action_space = MultiBinary(12)
        self.reward_coeff = 0.5
        self.full_hp = 176
        self.env  = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis', use_restricted_actions=retro.Actions.FILTERED)
        #state='guile', state='ryu'
        
        # Initialize health and score tracking
        self.prev_player_health = self.full_hp
        self.prev_oppont_health = self.full_hp
        self.score = 0
        
    def step(self, action):    
        obs, reward, done, info = self.env.step(action)
        obs = self.preprocess(obs) 

        #print(f"Step Observation Shape: {obs.shape}, Type: {obs.dtype}")
        # Frame delta 
        
       # frame_delta = (obs - self.previous_frame) #/255.0
        #frame_delta = np.clip(frame_delta, -1, 1)
        self.previous_frame = obs 
        
        
        frame_delta = obs 
        reward = info['score'] - self.score 
        self.score = info['score']
       # reward = self.calculate_custom_reward(info, reward)
        return frame_delta, reward, done, info    
    
    def render(self, *args, **kwargs):
        self.env.render()

    def reset(self):
        self.previous_frame = np.zeros((84, 84, 1), dtype=np.uint8)
        obs = self.env.reset()
        obs = self.preprocess(obs)

        # Reset tracking variables
        self.previous_frame = obs
        self.score = 0
        self.prev_player_health = self.full_hp
        self.prev_oppont_health = self.full_hp

        return obs
    
    def preprocess(self, obs):
        # Resize and convert to grayscale
        gray = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_CUBIC)  
        # Return with the correct shape and type #np.expand_dims(resize, axis=-1).astype(np.uint8)
        processed_obs = np.reshape(resize, (84,84,1))  #np.reshape(resize, (84,84,1))
        #print(f"Processed Obs Shape: {processed_obs.shape}, Type: {processed_obs.dtype}")
        return processed_obs
    
    
    
    def calculate_custom_reward(self, info, reward):
        # Safeguard for health values
        curr_player_health = max(0, min(info.get('health', self.full_hp), self.full_hp))
        curr_oppont_health = max(0, min(info.get('enemy_health', self.full_hp), self.full_hp))

        # Score-based reward
        reward += info.get('score', 0) - self.score
        self.score = info.get('score', 0)

        # Victory and loss bonuses
        if curr_oppont_health <= 0:
            reward += self.reward_coeff * 5000  # Victory bonus
        elif curr_player_health <= 0:
            reward -= self.reward_coeff * 2500  # Loss penalty

        # Ongoing reward during the fight
        else:
            reward += self.reward_coeff * (self.prev_oppont_health - curr_oppont_health)
            reward -= self.reward_coeff * (self.prev_player_health - curr_player_health)

        # Update health tracking
        self.prev_player_health = curr_player_health
        self.prev_oppont_health = curr_oppont_health

        return reward

    def close(self):
        self.env.close()
    