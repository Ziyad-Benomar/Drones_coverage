import numpy as np
import pygame
import gym
from math import floor, ceil


class Drone:
    def __init__(self, x, y, zone, max_distance):
        """
        # TODO: function documentation
        """
        self.position = np.array([x, y], dtype=np.float32)  # continuous position np.array(float)
        self.cell = np.zeros((2, ), dtype=np.int8)
        self.zone = zone  # the zone where the drone can move (numpy matrix)
        self.max_distance = max_distance  # TODO Mehdi je sais pas quelle valeur mettre ici, il ne faut pas que ça soit trop
        self.action_space = gym.spaces.Box(low=np.array([-1, 1]), high=np.array([-1, 1]), shape=(2, ))
        self.vision_field = []
        self.covered_zone = np.zeros_like(zone)  # 1 means that the cell is covered, 0 means that it is not
        self.last_observation = []
        self.delta_position = np.zeros(2)
        self.update()


    def get_grid_position(self):
        x, y = self.position
        height, width = self.zone.shape
        return min(int(floor(x)), width - 1), min(int(floor(y)), height - 1)

    def update_action_space(self):
        """
        Update the space of possible actions for the drone given its current state.

        The drone can move continuously on the grid but is not allowed to penetrate obstacles or to exit the grid.
        """
        eps = 0.0001
        height, width = self.zone.shape
        x, y = self.position
        # different results depending on the first side explored => maximize area ?
        max_combination = None
        max_area = 0
        for first_side in range(4):
            xmin = max(-self.max_distance, -x + eps)
            ymin = max(-self.max_distance, -y + eps)
            xmax = min(self.max_distance, width - x - eps)
            ymax = min(self.max_distance, height - y - eps)
            side = first_side
            for _ in range(4):
                side = (side + 1) % 4
                idx_to_check = []

                # Horizontal case, we need to explore all the vertical obstacles
                y_idx_to_check = [z for z in range(floor(y + ymin), min(int(floor(y + ymax)) + 1, height))]
                # Vertical case, we need to explore all the horizontal obstacles
                x_idx_to_check = [z for z in range(floor(x + xmin), min(int(floor(x + xmax)) + 1, width))]

                # Right
                if side == 0 and floor(x + xmax) > floor(x):
                    for i in y_idx_to_check:
                        if self.zone[i, floor(x + xmax)] < 0:
                            xmax = floor(x + xmax) - x - eps
                            break

                # Left
                if side == 2 and floor(x + xmin) < floor(x):
                    for i in y_idx_to_check:
                        if self.zone[i, floor(x + xmin)] < 0:
                            xmin = ceil(xmin + x) - x + eps
                            break

                # Up
                if side == 1 and floor(y + ymin) < floor(y):
                    for i in x_idx_to_check:
                        if self.zone[floor(y + ymin), i] < 0:
                            ymin = ceil(ymin + y) - y + eps
                            break

                # Down
                if side == 3 and floor(ymax + y) > floor(y):
                    for i in x_idx_to_check:
                        if self.zone[floor(ymax + y), i] < 0:
                            ymax = floor(ymax + y) - y - eps
                            break

            area = (xmax - xmin) * (ymax - ymin)
            if area > max_area:
                max_area = area
                max_combination = [xmin, xmax, ymin, ymax]

        xmin, xmax, ymin, ymax = max_combination
        self.action_space = gym.spaces.Box(low=np.array([xmin, ymin]), high=np.array([xmax, ymax]), shape=(2, ))

    # TODO : define a rule for the vision field
    # Here I suppose that the drone can see the eight cells around it
    def update_vision_field(self):
        x, y = self.get_grid_position()
        self.vision_field = []
        for dx in [-1, 0, 1]:
            if 0 <= x + dx < self.zone.shape[1]:
                for dy in [-1, 0, 1]:
                    if 0 <= y + dy < self.zone.shape[0]:
                        self.vision_field.append((x + dx, y + dy))

    # updating the covered_zone and last_observation
    def update_covered_zone(self):  # must be called after updating the vision_field
        self.last_observation = []
        for c in self.vision_field:
            if self.covered_zone[c[1], c[0]] == 0:
                self.last_observation.append(c)
                self.covered_zone[c[1], c[0]] = 1

    def update(self):
        self.update_action_space()
        self.update_vision_field()
        self.update_covered_zone()

    def move(self, action): # The action here is (delta_x,delta_y) and not the position !!
        self.position += action
        self.delta_position = action
        self.update()
    
    def greedy_mouvement(self, eps=0.005) :
        if np.random.rand() > eps and self.action_space.contains(self.delta_position) :
            return self.delta_position
        return self.delta_position.sample()






class Grid:
    def __init__(self, zone, drones_coords):
        self.zone = zone
        self.N = 0
        self.drones = []
        self.action_space = []  # list of action spaces of all the drones
        self.covered_zone = np.zeros_like(
            zone)  # 2 means that a drone is in the cell, 1 means that it is covered, 0 means that it is not
        self.found_targets = []
        self.last_observation = []
        self.num_targets = 0
        for i in range(self.zone.shape[0]) :
            for j in range(self.zone.shape[1]) :
                if self.zone[i,j] == 1 :
                    self.num_targets += 1
        # drawing attributes
        self.screen = None
        n = self.zone.shape[0]
        m = self.zone.shape[1]
        max_height = 900
        max_width = 1600
        max_cell_size = 100
        self.cell_size = int(np.array([max_height / n, max_width / m, max_cell_size]).min())
        # Adding drones
        for c in drones_coords:
            self.add_drone(c[0], c[1])
        self.update()


    # adds a drone in position x,y but does not update the grid attributes
    def add_drone(self, x, y):
        dr = Drone(x, y, self.zone, self.cell_size/256)
        self.drones.append(dr)
        self.N += 1

    def update_action_space(self):
        self.action_space = []
        for dr in self.drones:
            self.action_space.append(dr.action_space)


    # A cell covered by any drone is considered to be covered in the grid
    def update_covered_zone(self):
        for i in range(self.zone.shape[0]):
            for j in range(self.zone.shape[1]):
                if self.covered_zone[i, j] != 1:
                    for dr in self.drones:
                        if dr.covered_zone[i, j] == 1:
                            self.covered_zone[i, j] = 1
                            break
        for dr in self.drones:  # update covered_zone known by the drones also
            dr.covered_zone = self.covered_zone.copy()
        for dr in self.drones:  # locate the drones
            x, y = dr.get_grid_position()
            self.covered_zone[y, x] = 2

            # Last observed cells by all drones

    def update_last_observation(self):
        self.last_observation = []
        for dr in self.drones:
            for obs in dr.last_observation:
                if obs not in self.last_observation:
                    self.last_observation.append(obs)

    # Found targets after the last observations update
    def update_found_targets(self):  # assumes that update_last_observation has been called before
        for obs in self.last_observation:
            x = obs[0]
            y = obs[1]
            if self.zone[y, x] == 1:
                self.found_targets.append(obs)

    def update(self):  # Assumes that the drones attributes are updated
        self.update_action_space()
        self.update_covered_zone()
        self.update_last_observation()
        self.update_found_targets()
        if self.screen is not None:
            pygame.display.update()

    def move(self, action):
        for k in range(self.N):
            self.drones[k].move(action[k])
        self.update()
        self.draw()
    
    def move_position(self, position):
        for k in range(self.N):
            drone = self.drones[k]
            drone.move(position[k]-drone.position)
        self.update()
        self.draw()


    def draw_zone(self, lt=1):  # lt : line thickness
        cell_size = self.cell_size
        col_map = {0: (255, 255, 255), -1: (50, 50, 50), 1: (0, 255, 0)}
        n = self.zone.shape[0]
        m = self.zone.shape[1]
        for i in range(n):
            for j in range(m):
                x = j * cell_size
                y = i * cell_size
                pygame.draw.rect(self.screen, col_map[self.zone[i, j]],
                                 (x + lt, y + lt, cell_size - lt, cell_size - lt))

    def draw_drones(self):
        cell_size = self.cell_size
        radius = cell_size / 10
        for dr in self.drones:
            x, y = dr.position
            x = int(x * cell_size)
            y = int(y * cell_size)
            pygame.draw.circle(self.screen, (255, 255, 255), (x, y), radius+3)
            pygame.draw.circle(self.screen, (0, 0, 0), (x, y), radius)

    def draw_covered_zone(self, lt=2):  # lt :thickness
        cell_size = self.cell_size
        for i in range(self.zone.shape[0]):
            for j in range(self.zone.shape[1]):
                if self.covered_zone[i, j] == 0:
                    x = j * cell_size
                    y = i * cell_size
                    pygame.draw.lines(self.screen, (255, 0, 0), False, [(x, y), (x + cell_size, y + cell_size)], lt)
                    pygame.draw.lines(self.screen, (255, 0, 0), False, [(x + cell_size, y), (x, y + cell_size)], lt)

    def draw_last_observation(self, lt=3):  # lt :thickness
        cell_size = self.cell_size
        r = cell_size / 2
        for obs in self.last_observation:
            x = (obs[0] + 0.5) * cell_size
            y = (obs[1] + 0.5) * cell_size
            pygame.draw.lines(self.screen, (255, 0, 255), False, [(x - r, y), (x + r, y)], lt)
            pygame.draw.lines(self.screen, (255, 0, 255), False, [(x, y - r), (x, y + r)], lt)

    
    def draw_vision_field(self, lt=3) :
        for dr in self.drones :
            vf = dr.vision_field
            vf_x = np.array([c[0] for c in vf])
            vf_y = np.array([c[1] for c in vf])
            left = vf_x.min()*self.cell_size
            top = vf_y.min()*self.cell_size
            width  = (vf_x.max() - vf_x.min() + 1)*self.cell_size
            height = (vf_y.max() - vf_y.min() + 1)*self.cell_size
            pygame.draw.rect(self.screen, (255, 0, 0), pygame.Rect(left, top, width, height), 7)

    def draw_action_space(self, lt=3):
        cell_size = self.cell_size
        for dr in self.drones:
            x, y = dr.position
            x = int(x * cell_size)
            y = int(y * cell_size)
            arrow_radius = cell_size / 8
            left = x + int(dr.action_space.low[0] * cell_size)
            top = y + int(dr.action_space.low[1] * cell_size)
            width = int((dr.action_space.high[0] - dr.action_space.low[0]) * cell_size)
            height = int((dr.action_space.high[1] - dr.action_space.low[1]) * cell_size)
            pygame.draw.rect(self.screen, (0, 0, 255), pygame.Rect(left, top, width, height), 5)

    def draw(self):
        # creating the screen
        initial = (self.screen is None)
        if initial :
            pygame.init()
            screen_width = self.cell_size * self.zone.shape[1]
            screen_height = self.cell_size * self.zone.shape[0]
            self.screen = pygame.display.set_mode((screen_width, screen_height))
        self.screen.fill((0, 0, 0))  # fill screen with black

        self.draw_zone()
        self.draw_covered_zone()
        self.draw_last_observation()
        self.draw_action_space()
        self.draw_vision_field()
        self.draw_drones()
        pygame.display.update()

