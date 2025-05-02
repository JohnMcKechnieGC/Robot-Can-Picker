import numpy as np
from random import random, randint, seed
from Problem_Domain.action import Action


class Environment:
    LENGTH = 10
    CAN_PROBABILITY = 0.5
    REWARD_FOR_PICKING_UP_CAN = 10
    PENALTY_FOR_PICKING_UP_NOTHING = -1
    PENALTY_FOR_HITTING_A_WALL = -5

    def __init__(self, random_seed=None, record_actions=False):
        self.grid = None
        self.robot = None
        self.actions = []
        self.record_actions = record_actions
        self.reward = 0
        self.rewards = []
        self.randomise(random_seed)

    def randomise(self, random_seed=None):
        if random_seed is not None:
            seed(random_seed)
        self.grid = \
            np.array([[0 if random() < 1 - Environment.CAN_PROBABILITY else 1
                       for _ in range(Environment.LENGTH)]
                       for _ in range(Environment.LENGTH)])
        self.robot = None
        self.actions = []

    def printable_grid(self):
        grid = np.array(self.grid)
        if self.robot is not None:
            grid[self.robot.x][self.robot.y] += 3
        grid = np.transpose(grid)
        return grid
    
    def recent_rewards(self, n=10):
        if len(self.rewards) < n:
            return self.rewards
        return self.rewards[-n:]

    def set_robot(self, robot):
        self.robot = robot
        robot.x = 0
        robot.y = 0
        robot.score = 0

    def number_of_cans(self):
        return np.count_nonzero(self.grid == 1)

    def pick_up_can(self):
        if self.grid[self.robot.x][self.robot.y] == 1:
            self.reward = Environment.REWARD_FOR_PICKING_UP_CAN
            if self.record_actions:
                self.actions.append('pick up can')
        else:
            self.reward = Environment.PENALTY_FOR_PICKING_UP_NOTHING
            if self.record_actions:
                self.actions.append('pick up nothing')
        self.grid[self.robot.x][self.robot.y] = 0

    def move_north(self):
        if self.robot.y == 0:
            self.reward = Environment.PENALTY_FOR_HITTING_A_WALL
            if self.record_actions:
                self.actions.append(f'hit north wall ({self.robot.x}, {self.robot.y})')
        else:
            self.reward = 0
            self.robot.y -= 1
            if self.record_actions:
                self.actions.append(f'move up ({self.robot.x}, {self.robot.y})')

    def move_south(self):
        if self.robot.y == Environment.LENGTH - 1:
            self.reward = Environment.PENALTY_FOR_HITTING_A_WALL
            if self.record_actions:
                self.actions.append(f'hit south wall ({self.robot.x}, {self.robot.y})')
        else:
            self.reward = 0
            self.robot.y += 1
            if self.record_actions:
                self.actions.append(f'move down ({self.robot.x}, {self.robot.y})')

    def move_west(self):
        if self.robot.x == 0:
            self.reward = Environment.PENALTY_FOR_HITTING_A_WALL
            if self.record_actions:
                self.actions.append(f'hit west wall ({self.robot.x}, {self.robot.y})')
        else:
            self.reward = 0
            self.robot.x -= 1
            if self.record_actions:
                self.actions.append(f'move left ({self.robot.x}, {self.robot.y})')

    def move_east(self):
        if self.robot.x == Environment.LENGTH - 1:
            self.reward = Environment.PENALTY_FOR_HITTING_A_WALL
            if self.record_actions:
                self.actions.append(f'hit east wall ({self.robot.x}, {self.robot.y})')
        else:
            self.reward = 0
            self.robot.x += 1
            if self.record_actions:
                self.actions.append(f'move right ({self.robot.x}, {self.robot.y})')

    def move_random(self):
        match randint(0, 3):
            case 0:
                self.move_north()
            case 1:
                self.move_south()
            case 2:
                self.move_west()
            case 3:
                self.move_east()

    def do_nothing(self):
        self.reward = 0
        if self.record_actions:
            self.actions.append('do nothing')

    def perform_action(self, action):
        match action:
            case Action.pick_up_can:
                self.pick_up_can()
            case Action.move_north:
                self.move_north()
            case Action.move_south:
                self.move_south()
            case Action.move_west:
                self.move_west()
            case Action.move_east:
                self.move_east()
            case Action.move_random:
                self.move_random()
            case Action.do_nothing:
                self.do_nothing()
        self.robot.score += self.reward
        self.rewards.append(self.reward)
