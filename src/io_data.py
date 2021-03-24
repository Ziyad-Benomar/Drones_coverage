import numpy as np

from .hyperparameters import *
from .hyperparameters import HPHolder


class IOData(HPHolder):
    pass

class ContinuousState(IOData):
    def __init__(self, hp: HyperParameters):
        super().__init__(hp)
        self.type = 0
        self.P = np.zeros((self.hp.N, self.hp.space_dim))
        self.V = np.zeros((self.hp.N, self.hp.space_dim))
        self.T = 0

    def copy(self):
        result = ContinuousState(self.hp)
        result.type = self.type
        result.P = np.copy(self.P)
        result.V = np.copy(self.V)
        result.T = self.T
        return result

class DiscreteState(IOData):
    INFO_MAP_DIM = 2

    def __init__(self, hp: HyperParameters):
        super().__init__(hp)
        self.information_map = np.zeros((self.hp.k, self.hp.l), dtype=int)
        self.IR_map = np.zeros((self.hp.k, self.hp.l), dtype=bool)
        self.gamma_coords = np.zeros((self.hp.N, DiscreteState.INFO_MAP_DIM), dtype=int)

class DiscreteAction(IOData):
    def __init__(self, hp: HyperParameters):
        super().__init__(hp)
        self.joint_actions = np.zeros((self.hp.N, DiscreteState.INFO_MAP_DIM), dtype=int)

class Reward(IOData):
    def __init__(self, hp: HyperParameters):
        super().__init__(hp)
        self.reward = np.zeros((self.hp.N,))

class ContinuousAction(IOData):
    def __init__(self, hp: HyperParameters):
        super().__init__(hp)
        self.U = np.zeros((self.hp.N, self.hp.space_dim))

class DummyHusky:
    def __init__(self, i):
        self.i = i

    def get_altitude(self):
        return 10

    def faces_wall(self):
        return False

    def set_cmd_vel(self, val):
        pass

    def get_position(self):
        return np.array([0,0,0])

class AllAgentsFacade(HPHolder):
    def __init__(self, hp: HyperParameters, grid):
        super().__init__(hp)
        self.grid = grid
        self.pos 

    def register_pos():
        pass
        

class SimpleAgentFacade(HPHolder):
    def __init__(self, hp: HyperParameters, grid, continuous_state: ContinuousState, i):
        super().__init__(hp)
        self.grid = grid
        self.i = i
        self.continuous_state = continuous_state

    def get_altitude(self):
        return 0

    # def faces_wall(self):
    #     v = self.continuous_state.V[self.i, :]
    #     faces, distance, _ = self.underlying.faces_wall(self.hp.wall_detection_angle)
    #     return faces and distance < self.hp.wall_avoidance_distance

    def set_position(self, val):
        self.grid.drones[self.i].move(np.array(val))

    def get_position(self):
        drone = self.grid.drones[self.i]
        return np.array(drone.x, drone.y)

class PhysicalAgents(HPHolder):
    def __init__(self, hp: HyperParameters, grid, continuous_state: ContinuousState):
        super().__init__(hp)
        self.ros = [SimpleAgentFacade(hp, i, grid, continuous_state) for i in range(self.hp.N)]

    def get_altitudes(self):
        return np.array([agent.get_altitude() for agent in self.ros])
    
    # def get_walls(self):
    #     return np.array([agent.faces_wall() for agent in self.ros])

class Context(IOData):
    def __init__(self, hp: HyperParameters, env, grid):
        super().__init__(hp)
        self.continuous_state = ContinuousState(hp)
        self.reward = Reward(hp)
        self.discrete_state = DiscreteState(hp)
        self.discrete_action = DiscreteAction(hp)
        self.continuous_action = ContinuousAction(hp)
        self.grid = grid
        self.env = env
        self.physical_agents = PhysicalAgents(hp, grid, self.continuous_state)
        self.done = False

    def update_continuous(self):
        dim = self.hp.space_dim
        for num, agent in enumerate(self.physical_agents.ros):
            self.continuous_state.P[dim*num:dim*(num+1)] = agent.get_position()

    def apply_velocity_command(self, V):
        dim = self.hp.space_dim
        for num, agent in enumerate(self.physical_agents.ros):
            agent.set_cmd_vel(V[dim*num:dim*(num+1)])

class ContextHolder(HPHolder):
    def __init__(self, hp: HyperParameters, context: Context):
        super().__init__(hp)
        self.context = context
