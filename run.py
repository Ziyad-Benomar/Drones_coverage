import sys
import time

import gym
import numpy as np
import pygame

import matplotlib.pyplot as plt

import src.covering as covering
from src.hyperparameters import HyperParameters
from src.complete_framework import Framework





#######################################################################
#######################################################################
# CHOOSE THE PARAMETERS HERE
#######################################################################
#######################################################################

# dimentions of the map
width = 15
height = 10

# Probability of havinf an obstacle/human in a cell
p_obstacle = 0.15
p_human = 0.2

# number of drones
N = 5

# Number of episodes in the RL
n_episodes = 100000

# Max number of iterations in an episode
max_t = 1000
#######################################################################
#######################################################################








# Function to generate a random zone
def random_zone(n, m, drone_entries=[(0, 0)], p_obstacle=0.1, p_human=0.2):
    zone = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            if (j, i) in drone_entries:
                continue
            rd = np.random.rand()
            if rd < p_obstacle:
                zone[i, j] = -1
            elif rd < p_obstacle + p_human:
                zone[i, j] = 1
    return zone

def random_policy(env) :
    gr = env.grid
    action = []
    for dr in gr.drones :
        action.append(dr.action_space.sample())
    return action


def drone_greedy_policy(prev_action, action_space, last_obs, eps=0.3):
    # if we discovered new cells in a direction, it is likely to discover others by going in the same direction again
    rd = np.random.rand()
    if len(last_obs) > 0:
        rd *= len(last_obs)
    if action_space.contains(prev_action) and rd > eps :
        return prev_action
    """
    c'est cassé avec le passage au continu
    opp_prev_action = -prev_action  # the opposite of the previous action
    if action_space.contains(opp_prev_action):
        action_space.remove(opp_prev_action)
    """
    return action_space.sample()


def greedy_policy(env, prev_action, eps=0.3):
    gr = env.grid
    action = []
    for k in range(len(gr.drones)):
        dr_action_space = gr.drones[k].action_space
        for c in gr.drones[k].vision_field:
            x, y = gr.drones[k].get_grid_position()
            """


            ca aussi c'est cassé
            if not c == (x, y) and gr.covered_zone[c[1], c[0]] == 2:
                # a cell containing another drone
                rd_x = np.random.rand()
                rd_y = np.random.rand()
                if rd_x < 1 / 2:  # move away in the x axis
                    if c[0] > x and 0 in dr_action_space:
                        dr_action_space.remove(0)  # 0 : move right
                    elif c[0] < x and 2 in dr_action_space:
                        dr_action_space.remove(2)  # 2 : move left
                if rd_y < 1 / 2:  # move away in the y axis
                    if c[1] > y and 3 in dr_action_space:
                        dr_action_space.remove(3)  # 3 : move down
                    elif c[1] < y and 1 in dr_action_space:
                        dr_action_space.remove(1)  # 1 : move up
        

            """
#        et ça aussi :'(
#        if len(dr_action_space) == 0:
#            dr_action_space = gr.drones[k].action_space[:]
        action.append(drone_greedy_policy(prev_action[k], dr_action_space, gr.drones[k].last_observation, eps))
    return action


#Setting hyperparams of the environment
hp = HyperParameters(N, width, height)
drones_entries = []
for k in range(N):
    drones_entries.append((np.random.randint(width), np.random.randint(height))) #(x, y) tuples of initial positions of each drone

# Random zone creation
zone = random_zone(height, width, drones_entries, p_obstacle)

# GYM !!
i = 0
n_iter = hp.nb_sim
max_iter = hp.max_sim

limit_t = np.array([n_iter, max_iter, n_episodes]).min()

done_time = []

# loop for training the RL
while i < n_episodes:
    framework = Framework(zone, drones_entries, hp)
    framework.hp.n = height
    framework.hp.m = width
    framework.context.grid.draw()
    #time.sleep(1) # To see the initial positions of the drones

    if i!=0:
        framework.rl_module.load_models() # Load QNetwork checkpoint

    action = random_policy(framework.env)
    done_t = 0
    for _ in range(max_t):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
                break
        done_t += 1
        #time.sleep(0.03)  # Modify here the time between the iterations
        framework.run_step() 
        if framework.context.done:
            #while True:
                #time.sleep(0.2)
            break
    done_time.append(done_t)
    framework.rl_module.save_models() # Save QNetwork checkpoint
    framework.quit()
    i+=1
    if i%10 == 0 :
        print("Episode num ",i)
        print(done_time)



plt.plot(done_time)

plt.show()




    
    
    
