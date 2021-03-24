from gym.envs.registration import register
import numpy as np

register(
    id='covering-v0',
    entry_point='src.covering.envs:CoveringEnv',
    kwargs={'zone' : np.zeros((10,15)), 'drones_coords' : [(0,0)], 'hp' : None, 'context' : None, 'gamma_map' : None}
)
