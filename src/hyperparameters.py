import numpy as np

class HyperParameters:
    @staticmethod
    def compute_T_min(m, n, k, l, v_max):
        return min(((l-1)*m*k)/(l*np.abs(v_max)) + ((k-1)*n)/(k*np.abs(v_max))
                    , ((k-1)*n*l)/(k*np.abs(v_max)) + ((l-1)*m)/(l*np.abs(v_max)))

    def __init__(self, N, width, height):
        self.N = N  # The Number of agents
        self.N_T = 100000  # The maximum training times

        self.max_sim = 10000 # max number of simulations to run
        self.nb_sim = 100 # number of simulations to run

        self.space_dim = 2
        self.gamma_map_dim = 2

        # Size of the target area
        self.m = width
        self.n = height

        # Size of the γ-information map
        self.k = width
        self.l = height

        # Dimensions of the γ-information map
        self.cell_width = self.m / self.k
        self.cell_height = self.n / self.l

        self.alpha = 1  # Learning rate
        self.lambda_ = 0.8  # Discounting factor
        self.w = 0.5  # Weight of the cooperative learning
        self.h = 0.1  # Parameters of the bump function
        self.r_ref = 64  # Standard reward value for the whole traversal process
        self.v_max = 1  # Magnitude of the maximum velocity of the agent   1m.s lol
        self.T_min = HyperParameters.compute_T_min(self.m, self.n, self.k, self.l, self.v_max)
        self.r_s = None  # perceived radius
        self.r_c = None  # Communication distance
        self.epsilon = 0.2  # Permissible position error
        self.v_max = 1  # Magnitude of the maximum velocity of the agent   1m.s lol

        self.c_r = 0.5
        self.c_1_T = 0.5
        self.c_2_T = 0.5

        # Control Parameters of the MAS motion model
        self.c_alpha = 0.5
        self.c_1_gamma = 0.5
        self.c_2_gamma = 0.8

        self.d = 8  # Avoidance distance

        self.wall_detection_angle = 1 #rad
        self.wall_avoidance_distance = 1 #m

        self.integration_span = [0,1]  # Time step for the integration of the differential equations for action-to-state derivation

        self.stationary_altitude = 1
        self.altitude_adjutsment_intensity = 0.1
        self.altitude_margin = 1

        def __str__(self):
            return '\n'.join(f'{name} = {value}' for name, value in self.__dict__.items())

class HPHolder:
    def __init__(self, hp : HyperParameters):
        self.hp = hp
