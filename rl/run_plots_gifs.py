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
    main_path = "./data/"
    img_path = "image2.png"
    msk_path = 'mask2.png'

    load_image, load_mask = load_image_mask(main_path, img_path, msk_path)


    os.makedirs("models/", exist_ok=True)
    os.makedirs("gifs/", exist_ok=True)
    os.makedirs("plots/", exist_ok=True)


    env = ev.Mask_Proj_Env(load_image, load_mask)



    #######################################################
    ########### run
    #######################################################
    q_file = 'models/antagonist'
    agent_info = {"num_actions": env.nA, "num_states": 4, "step_size": 0.01, "still": 0, "path": q_file}
    num_episodes = 50 # The number of episodes in each run
    n_steps=10000
    agent = ag.TabularSoftmaxQWindowAgent()

    single_run=[]
    agent.agent_init(agent_info)
    old_ep_cost=0
    #ciclo su episodi
    for episode in tqdm(range(1, num_episodes+1)):
        # Runs an episode
        images = []
        done = False
        rewards=[]
        reward = env.reset()
        action = agent.agent_start(reward)
                
        if episode%5==0:
            images.append(env.render(True))
            rewards.append(reward)
        for step in range(n_steps):
            reward, done, info = env.step(action)
            if episode%5==0:
                images.append(env.render(True))
                rewards.append(reward)
            if done:
                manhattan = int(info["Manhattan"])
                cost=np.round(n_steps/manhattan)
                single_run.append(cost)
                agent.agent_end(episode)
                
                if episode%5==0:
            
                    #save gif
                    imageio.mimsave('gifs/result'+str(episode)+'.gif', images[:], fps=10)
                
                    # plot results
                    plt.plot(single_run, label="Greedy Softmax Q-window")
                    plt.xlabel("Episodes")
                    plt.ylabel("Normalized\nsteps\nduring\nepisode", labelpad=40)
                    plt.xlim(0,len(single_run))
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig('plots/plots_'+str(episode)+'.png')
                    plt.gcf().clear()
                
                
                    plt.plot(rewards, label="Rewards at each step")
                    plt.xlabel("Steps")
                    plt.ylabel("Reward",rotation=0, labelpad=40)
                    plt.xlim(0,len(rewards))
                    plt.axvline(manhattan, color='r', label = "Minimum number of actions")
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig('plots/plot_reward_'+str(episode)+'.png')
                    plt.gcf().clear()
                        
                
                break
            action = agent.agent_step(reward)  

        if not done:
            manhattan = int(info["Manhattan"])
            cost=np.round(n_steps/manhattan)
            old_ep_cost=cost
            single_run.append(cost)
            if episode%5==0:
                #save gif
                imageio.mimsave('gifs/result'+str(episode)+'.gif', images[:], fps=10)
            
                # plot results
                plt.plot(single_run, label="Greedy Softmax Q-window")
                plt.xlabel("Episodes")
                plt.ylabel("Normalized\nsteps\nduring\nepisode", labelpad=40)
                plt.xlim(0,len(single_run))
                plt.legend()
                plt.tight_layout()
                plt.savefig('plots/plots_'+str(episode)+'.png')
                plt.gcf().clear()
            
            
                plt.plot(rewards, label="Rewards at each step")
                plt.xlabel("Steps")
                plt.ylabel("Reward",rotation=0, labelpad=40)
                plt.xlim(0,len(rewards))
                plt.axvline(manhattan, color='r', label = "Minimum number of actions")
                plt.legend()
                plt.tight_layout()
                plt.savefig('plots/plot_reward_'+str(episode)+'.png')
                plt.gcf().clear()
        #######################################################
                
        
        

        # single_run=[]
        # #agent.agent_init(agent_info, True)

        # images = []
        # rew=[]
        # done = False
        # # Runs an episode
        # old_rew = env.reset()
        # rew.append(old_rew)    
        # img = env.render()
        # images.append(img)
        # action = agent.agent_start(old_rew)

        # for step in tqdm(range(int(n_steps))):
            # #print("Action: {}".format(env.steps[action]))

            # reward, done, _ = env.step(action)
            # rew.append(reward)    
            # img = env.render(True)
            # images.append(img)
            # if done:
                # break
            # action = agent.agent_step(reward) 
         
        # imageio.mimsave('gifs/result'+str(i)+'.gif', images[:], fps=10)

        # ########################################################
        
        
        # plt.plot(rew, label="Rewards at each step")
        # plt.xlabel("Steps")
        # plt.ylabel("Reward",rotation=0, labelpad=40)
        # plt.xlim(0,len(img))
        # plt.legend()
        # #plt.show()
        # plt.savefig('plots/plots_single'+str(i)+'.png')



if __name__ == "__main__":
    main()
