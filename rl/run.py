import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
import time
import os, sys, random
from pathlib import Path
import imageio
import Mask_Proj_Env as ev
import TabularSoftmaxQWindowAgent as ag
from tqdm import tqdm

def load_image_mask(path, img_path, msk_path):
    """ Load image and mask (in final prototype will be received from previous step in pipeline)

    Parameters:
        path: relative path to folder with image and mask files
        img_path: image file name (with extension)
        msk_path: mask file name (with extension)

    Returns:
        image: image loaded
        mask: mask loaded
    """
    #print(os.path.join(path, img_path))
    image = cv2.imread(os.path.join(path, img_path))
    mask = cv2.cvtColor(cv2.imread(os.path.join(path, msk_path)), cv2.COLOR_BGR2GRAY)

    thrshd = 100  ### to delete artifacts
    mask[mask > thrshd] = 255
    mask[mask <= thrshd] = 0
    
     
    return image, mask

def main():
    train = False
    load  = False
    save  = False
    gif   = False
    #train, load, save, gif
    print(len(sys.argv))
    for str in sys.argv:
        print(str)
        s = str.split("=")
        if s[0] == "train":
            train = s[1] == "True"
        elif s[0] == "load":
            load = s[1]
        elif s[0] == "save":
            load = s[1]
        elif s[0] == "gif":
            gif = s[1]            
    
    main_path = "../data/"
    img_path = "crop.jpg"
    msk_path = 'mask.jpg'

    load_image, load_mask = load_image_mask(main_path, img_path, msk_path)

    if(load or save):
        os.makedirs("models/", exist_ok=True)
    if(gif):
        os.makedirs("gifs/", exist_ok=True)

    #print("Init Env")
    env = ev.Mask_Proj_Env(load_image, load_mask)


    agent_info = {"num_actions": env.nA, "num_states": 4, "step_size": 0.001, "still": 0, "path": load}
    agent = ag.TabularSoftmaxQWindowAgent()
    agent.agent_init(agent_info)
    single_run=[]
    n_steps=20000 # number of maximum steps per episode
    
    if train:
        #######################################################
        ########### run train
        #######################################################    
        num_episodes = 100   # number of episodes
        old_ep_cost = 0

        for episode in tqdm(range(1, num_episodes+1)):
            # Runs an episode
            done = False
            rewards=[]

            reward = env.reset()
            action = agent.agent_start(reward)
                    
            for step in range(n_steps):
                reward, done, info = env.step(action)
                rewards.append(reward)
                    
                # if the episode ended in the proper alignment
                if done:                         
                    break
                action = agent.agent_step(reward)  

            # if the episode ended NOT in the proper alignment
            #if not done:
            single_run.append(sum(rewards))

            #######################################################

        if (save):
            agent.agent_save(save)
            
            
    if (gif):
    
        single_run=[]

        images = []
        rewards=[]
        done = False
        
        # Runs an episode
        reward = env.reset()
        action = agent.agent_start(reward)
        images.append(env.render(True))
        rewards.append(reward)
        for step in tqdm(range(int(n_steps))):
            reward, done, info = env.step(action)
            images.append(env.render(True))
            rewards.append(reward)
            if done:
                break
            action = agent.agent_step(reward) 
        
        imageio.mimsave("gifs/"+gif, images[:], fps=10)
        


if __name__ == "__main__":
    main()
