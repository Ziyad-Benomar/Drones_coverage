import math
from os import stat_result
from typing import Tuple

import numpy as np
import scipy.integrate

from .agent import DuelingDQNAgent
from .hyperparameters import *
from .io_data import *
from .model import TransformedDiscreteState


class IONode(ContextHolder):
    pass


ALL = None


def to_gamma_map_coords_single(x, y, m, n, k, l):
    eps = 10e-8
    x = abs(x - eps)
    y = abs(y - eps)
    return math.floor((x / m) * k), math.floor((y / n) * l)


to_gamma_map_coords = np.vectorize(to_gamma_map_coords_single, excluded=set([2, 3, 4, 5]))


class GammaInfoMap(IONode):
    def __init__(self, hp: HyperParameters, context: Context):
        super().__init__(hp, context)
        self.discrete_state = self.context.discrete_state

    def compute(self, continuous_state: ContinuousState, to_update=ALL) -> DiscreteState:
        m, n, k, l = self.hp.m, self.hp.n, self.hp.k, self.hp.l
        if to_update is not ALL:
            x = continuous_state.P[to_update, 0]
            y = continuous_state.P[to_update, 1]
            gamma_coords = np.array(to_gamma_map_coords(X, Y, m, n, k, l)).T
            self.discrete_state.gamma_coords[to_update, :] = gamma_coords
            self.discrete_state.information_map[gamma_coords[0], gamma_coords[1]] += 1
        else:
            X = continuous_state.P[:, 0]
            Y = continuous_state.P[:, 1]
            gamma_coords = np.array(to_gamma_map_coords(X, Y, m, n, k, l)).T
            self.discrete_state.gamma_coords[:, :] = gamma_coords
            self.discrete_state.information_map[gamma_coords[:, 0], gamma_coords[:, 1]] += 1

        return self.discrete_state

    def get_gamma_ij(self, coords: Tuple) -> int:
        return self.discrete_state.information_map[coords[0], coords[1]]

    def is_totally_visited(self) -> bool:
        for i in range(0, self.hp.k):
            for j in range(0, self.hp.l):
                if self.discrete_state.information_map[i, j] == 0:
                    return False
        return True


class Environment(HPHolder):
    def __init__(self, hp: HyperParameters):
        super().__init__(hp)

        class ObsSpace:
            def __init__(self):
                self.shape = (hp.k, hp.l)

        class ActionSpace:
            def __init__(self):
                self.n = 9

        self.observation_space = ObsSpace()
        self.action_space = ActionSpace()


class RLModule(IONode):
    def __init__(self, hp: HyperParameters, context: Context):
        super().__init__(hp, context)
        self.load_checkpoint = False
        env = Environment(hp)
        self.agents = [DuelingDQNAgent(hp,context,
                                       env.action_space.n,
                                       TransformedDiscreteState.Shape.from_hp(hp),
                                       mem_size=50000,
                                       batch_size=32,
                                       drone_id=i,
                                       replace=10000,
                                       chkpt_dir='models/')
                       for i in range(self.hp.N)]

        for agent in self.agents:
            if self.load_checkpoint:
                agent.load_models()

        self.state = None
        self.scores = np.zeros(self.hp.N)

    def compute(self, discrete_state: DiscreteState) -> DiscreteAction:
        self.state = discrete_state
        result = DiscreteAction(self.hp)

        for i, agent in enumerate(self.agents):
            result.joint_actions[i] = agent.choose_action(discrete_state)
        return result

    def train(self, discrete_state: DiscreteState, reward: Reward):
        state = self.state
        new_state = discrete_state
        ksis = np.array([agent.compute_ksi(state, new_state, reward) for agent in self.agents])

        for i in range(self.hp.N):
            self.scores[i] += reward.reward[i]

        if not self.load_checkpoint:
            for i, agent in enumerate(self.agents):
                agent.store_transition(state, agent.action, reward.reward[i], new_state, ksis[i])
                agent.learn()
        self.state = new_state
    
    def save_models(self):
        for agent in self.agents:
                agent.save_models()

    def load_models(self):
        self.load_checkpoint = True
        for agent in self.agents:
                agent.load_models()



class MASMotionModel(IONode):
    @staticmethod
    def sigma(z):
        return z / np.sqrt(1 + np.inner(z, z))

    @staticmethod
    def smooth_abs(z, norm=lambda z: z ** 2):
        return np.sqrt(1 + norm(z)) - 1

    @staticmethod
    def sigma_norm(z):
        return MASMotionModel.smooth_abs(z, norm=lambda z: np.inner(z, z))

    @staticmethod
    def get_rho(h):
        def rho(z):
            assert (z >= 0)
            return 1 if z < h else \
                0 if h >= 1 else \
                    (1 + np.cos(math.pi * (z - h) / (1 - h))) / 2

        return rho

    def __init__(self, hp: HyperParameters, context: Context):
        super().__init__(hp, context)
        self.rho_h = MASMotionModel.get_rho(self.hp.h)
        self.d_sigma = MASMotionModel.smooth_abs(self.hp.d)

    def phi(self, z):
        return self.rho_h(z / self.hp.d) * (MASMotionModel.sigma(z - self.d_sigma) - 1)

    def f_alpha(self, pi, pj):
        return self.phi(MASMotionModel.sigma_norm(pj - pi)) * MASMotionModel.sigma(pj - pi)

    def compute(self, continuous_state: ContinuousState, discrete_action: DiscreteAction) -> ContinuousAction:
        N = self.hp.N
        U_alpha = np.array(
            [sum(self.f_alpha(continuous_state.P[i], continuous_state.P[j]) for j in range(N) if j != i) for i in
             range(N)])
        U_alpha *= self.hp.c_alpha
        # N, d = discrete_action.joint_actions.shape
        # action_3d = np.zeros((N, d + 1))
        # action_3d[:, :-1] = discrete_action.joint_actions
        U_gamma_target = -self.hp.c_1_gamma * (continuous_state.P - discrete_action.joint_actions)
        U_gamma_velocity = -self.hp.c_2_gamma * continuous_state.V
        self.context.continuous_action.U = U_alpha + U_gamma_target + U_gamma_velocity
        return self.context.continuous_action

# class ContinuousActionTwister(IONode):
#     def __init__(self, hp: HyperParameters, context: Context):
#         super().__init__(hp, context)

#     def compute(self, continuous_action: ContinuousAction):
#         for i, (altitude, faces_wall) in enumerate(
#                 zip(self.context.physical_agents.get_altitudes(), self.context.physical_agents.get_walls())):
#             if faces_wall or altitude < self.hp.stationary_altitude - self.hp.altitude_margin / 2:
#                 continuous_action.U[i, :-1] = 0
#                 continuous_action.U[i, -1] = self.hp.altitude_adjutsment_intensity
#             elif altitude > self.hp.stationary_altitude + self.hp.altitude_margin / 2:
#                 continuous_action.U[i, -1] = -self.hp.altitude_adjutsment_intensity
#         return continuous_action


class Agent(IONode):
    def __init__(self, hp: HyperParameters, context: Context, gamma_map: GammaInfoMap):
        super().__init__(hp, context)
        self.continuous_state = self.context.continuous_state
        self.gamma_map = gamma_map
        self.reward = self.context.reward
        self.context = context

    def update_continuous_state(self, continuous_action: ContinuousAction):
        features = 2
        dim = self.hp.space_dim
        elem_size = dim * features
        self.context.env.reward_calculator.update()

        def get_diff_system(U):
            def func(t, Y):
                num_variables, = Y.shape
                assert (num_variables == elem_size * self.hp.N)
                Yp = np.zeros((num_variables,))
                for axis in range(dim):
                    Yp[axis::elem_size] = Y[dim + axis::elem_size]
                    Yp[dim + axis::elem_size] = U[axis::dim]
                return Yp

            return func

        Y0 = np.zeros((self.hp.N * elem_size,))
        for axis in range(dim):
            Y0[axis::elem_size] = self.continuous_state.P[:, axis]
            Y0[dim + axis::elem_size] = self.continuous_state.V[:, axis]
        updated = scipy.integrate.solve_ivp(get_diff_system(continuous_action.U.flatten()),
                                            self.hp.integration_span, Y0)
        finalY = updated.y[:, -1]
        for axis in range(dim):
            self.continuous_state.P[:, axis] = finalY[axis::elem_size]
            self.continuous_state.V[:, axis] = finalY[dim + axis::elem_size]

        global_action = []
        for drone_idx, position in enumerate(self.continuous_state.P):
            drone = self.context.env.grid.drones[drone_idx]
            action = position - drone.position
            if not drone.action_space.contains(action):
                xmin, ymin = drone.action_space.low
                xmax, ymax = drone.action_space.high
                x = max(xmin, min(xmax, action[0]))
                y = max(ymin, min(ymax, action[1]))
                action = np.array([x, y])
            global_action.append(action + drone.position)
        global_action = np.array(global_action)
        result = self.context.env.step(global_action)
        return result

    def compute(self, continuous_action: ContinuousAction) -> Tuple[ContinuousState, Reward]:
        observation, reward, done, info = self.update_continuous_state(continuous_action)
        self.reward.reward[:] = reward
        self.context.done = done

        return self.continuous_state, self.reward
