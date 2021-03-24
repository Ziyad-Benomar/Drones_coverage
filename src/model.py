import os
from math import gamma

import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .hyperparameters import HyperParameters


class TransformedDiscreteState:
    class Shape:
        def __init__(self, map_shape, n_drones):
            self.map_shape = map_shape
            self.n_drones = n_drones

        @staticmethod
        def from_hp(hp: HyperParameters):
            return TransformedDiscreteState.Shape((hp.k, hp.l), hp.N)

    def __init__(self, information_map, gamma_coords):
        self.information_map = information_map
        self.gamma_coords = gamma_coords

class QNetwork(nn.Module):

    """ Actor policy Model """

    def __init__(self, hp: HyperParameters, state_shape: TransformedDiscreteState.Shape, action_size, name, chkpt_dir, fc1_unit=64, fc2_unit = 64):

        super(QNetwork,self).__init__()
        self.hp = hp
        self.state_shape = state_shape

        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.fc1  = nn.Linear(state_shape.map_shape[0] * state_shape.map_shape[1] + state_shape.n_drones * hp.gamma_map_dim,fc1_unit)
        self.fc2  = nn.Linear(fc1_unit,fc2_unit)
        self.V    = nn.Linear(fc2_unit,1)
        self.A    = nn.Linear(fc2_unit,action_size)

        self.optimizer = optim.RMSprop(self.parameters(), lr=self.hp.alpha)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    @staticmethod
    def prepare_inputs(information_map, gamma_coords):
        return T.cat([T.flatten(information_map), T.flatten(gamma_coords)])

    @staticmethod
    def prepare_batched_inputs(information_map, gamma_coords):
        assert(information_map.shape[0] == gamma_coords.shape[0])
        return T.cat([T.flatten(information_map, start_dim=1), T.flatten(gamma_coords, start_dim=1)], dim=1)
    
    def forward(self, input):
        """
        We implement here the Dueling Q Network forward propagation.
        """
        fonction_1 = F.relu(self.fc1(input))
        fonction_2 = F.relu(self.fc2(fonction_1))

        V = self.V(fonction_2)
        A = self.A(fonction_2)
        return V,A

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
