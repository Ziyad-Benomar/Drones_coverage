import numpy as np

from .hyperparameters import HyperParameters
from .model import *

class ReplayBuffer(object):
    def __init__(self, hp: HyperParameters, max_size, state_shape:TransformedDiscreteState.Shape, action_size):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.info_map_mem = np.zeros((self.mem_size, *state_shape.map_shape), dtype = int)
        self.gamma_coords_mem = np.zeros((self.mem_size, state_shape.n_drones, 2), dtype = int)
        self.new_info_map_mem = np.zeros((self.mem_size, *state_shape.map_shape), dtype = int)
        self.new_gamma_coords_mem = np.zeros((self.mem_size, state_shape.n_drones, 2), dtype = int)

        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.ksi_memory = np.zeros((self.mem_size, state_shape.n_drones), dtype = float)

    def store_transition(self, state: TransformedDiscreteState, action, reward, new_state: TransformedDiscreteState, ksis):
        index = self.mem_cntr % self.mem_size
        self.info_map_mem[index] = state.information_map
        self.gamma_coords_mem[index] = state.gamma_coords
        self.info_map_mem[index] = new_state.information_map
        self.new_gamma_coords_mem[index] = new_state.gamma_coords
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.ksi_memory[index] = ksis
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = TransformedDiscreteState(self.info_map_mem[batch], self.gamma_coords_mem[batch])
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        new_states = TransformedDiscreteState(self.new_info_map_mem[batch], self.new_gamma_coords_mem[batch])
        ksis = self.ksi_memory[batch]

        return states, actions, rewards, new_states, ksis
