import sys

import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding

from .mods.grid import Grid
from ...io_data import *
from ...io_nodes import *

class RewardCalculator(ContextHolder):
    def __init__(self, hp: HyperParameters, context: Context, gamma_map: GammaInfoMap):
        super().__init__(hp, context)
        self.continuous_state = self.context.continuous_state
        self.prev_continuous_state = None
        self.gamma_map = gamma_map

    def update(self):
        self.prev_continuous_state = self.continuous_state.copy()  # update old position

    def compute_reward(self):
        m, n, k, l, v_max, c_1_T, c_2_T = self.hp.m, self.hp.n, self.hp.k, self.hp.l, self.hp.v_max, self.hp.c_1_T, self.hp.c_2_T
        reward = np.zeros((self.hp.N))
        for i in range(0, self.hp.N):
            coord = np.array(to_gamma_map_coords_single(self.continuous_state.P[i, 0],
                                                              self.continuous_state.P[i, 1], m, n, k, l))
            coord_prev = np.array(to_gamma_map_coords_single(self.prev_continuous_state.P[i, 0],
                                                                   self.prev_continuous_state.P[i, 1], m, n, k, l))

            delta_coord = ((coord - coord_prev)[0] + 1) + 3 * ((coord - coord_prev)[1] + 1)
            equivalent_map = [4, 3, 2, 5, 0, 1, 6, 7, 8]
            ai = equivalent_map[delta_coord]

            gamma = self.gamma_map.get_gamma_ij(coord)
            reward_i = None
            if gamma == 0 and ai in [0, 1, 3, 5, 7]:
                reward_i = 0
            elif gamma == 0 and ai in [2, 4, 6, 8]:
                reward_i = -0.3
            elif gamma != 0:
                reward_i = -0.2 * np.exp(self.hp.c_r * (gamma - 1))

            elif self.gamma_map.is_totally_visited():  # Rt est commum a tous, il est stocke dans l objet de classe gamma map
                self.hp.T_min = np.min((l - 1) * m * k / (l * np.abs(v_max)) + (k - 1) * n / (k * np.abs(v_max)),
                                       (l - 1) * m / (l * np.abs(v_max)) + (k - 1) * n * l / (k * np.abs(v_max)))
                if self.continuous_state.T > c_1_T * self.hp.T_min:
                    reward_i = 0

                elif self.continuous_state.T <= c_2_T * self.hp.T_min:
                    reward_i = self.hp.r_ref
                else:
                    reward_i = self.hp.r_ref / 2 * [1 + np.cos(
                        np.pi * (self.continuous_state.T - c_2_T * self.hp.T_min) / (
                                c_1_T * self.hp.T_min - c_2_T * self.hp.T_min))]

            assert(reward_i is not None)
            reward[i] = reward_i
        return reward


class CoveringEnv(gym.Env): 
    metadata = {'render.modes': ['human']}

    def __init__(self, zone, drones_coords, hp: HyperParameters, context: Context, gamma_map: GammaInfoMap): 
        self.grid = Grid(zone, drones_coords)
        self.zone = zone
        self.drones_initial_coords = drones_coords
        self.hp = hp
        self.context = context
        self.t_max = self.hp.N_T
        if self.t_max == 0 :
            self.tmax = self.zone.shape[0]*self.zone.shape[1]
        self.t = 0
        self.reward_calculator = RewardCalculator(hp, context, gamma_map)

    def step(self, action):
        # do the action
        self.grid.move_position(action)
        for i in range(self.hp.N):
            self.context.continuous_state.P[i, :] = self.grid.drones[i].position
        self.t += 1
        # observation
        observation = self.grid.last_observation
        # reward
        reward = self.reward_calculator.compute_reward()
        # done, stop when all the zone is covered ? or when all targets are found ? ...
        #done = (self.grid.covered_zone.flatten().prod() != 0 or self.t >= self.t_max)
        done = (len(self.grid.found_targets) == self.grid.num_targets or self.t >= self.t_max)
        # info
        info = {}
        return observation, reward, done, info

    def reset(self):
        if self.grid.screen != None :
            pygame.quit(); sys.exit()
        self.grid = Grid(self.zone, self.drones_initial_coords)

    def render(self, mode='human', close=False):
        pass # TODO ?
        
        
        
        
