import gym

from .io_nodes import *
from .covering.envs import CoveringEnv

class Framework:
    def __init__(self, zone, drones_coords, hp = HyperParameters(3,5,5)):
        self.hp = hp
        self.context = Context(self.hp, None, None)
        self.discretizer = GammaInfoMap(self.hp,self.context)
        self.agent = Agent(self.hp,self.context,self.discretizer)
        self.rl_module = RLModule(self.hp,self.context)
        self.mas = MASMotionModel(self.hp,self.context)
        #self.action_twister = ContinuousActionTwister(self.hp, self.context)
        self.env = CoveringEnv(zone, drones_coords, self.hp, self.context, self.discretizer)
        self.env = gym.make('covering-v0',
                            zone=zone,
                            drones_coords=drones_coords,
                            hp=self.hp,
                            context=self.context,
                            gamma_map=self.discretizer)
        self.context.env = self.env
        self.context.grid = self.env.grid
        for i in range(self.hp.N):
            self.context.continuous_state.P[i, :] = self.env.grid.drones[i].position
        self.env.reward_calculator.update()

    def run_step(self):
        context = self.context
        context.discrete_action = self.rl_module.compute(context.discrete_state) # 
        context.continuous_action = self.mas.compute(context.continuous_state, context.discrete_action) #
        #context.continuous_action = self.action_twister.compute(context.continuous_action)
        context.continuous_state, context.reward = self.agent.compute(context.continuous_action) # Get continuous state from the continuous action
        context.discrete_state = self.discretizer.compute(context.continuous_state) # Convert to discrete state for the RL
        self.rl_module.train(context.discrete_state, context.reward)
        self.env.render()

    def run_sim(self):
        context = self.context
        context.discrete_state = self.discretizer.compute(context.continuous_state)
        for i in range(self.hp.N_T):
            self.run_step()

    def quit(self):
        self.env.close()
