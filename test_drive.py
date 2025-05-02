import os
import time
from colorama import init, Back

from Problem_Domain.environment import Environment
from Robots.base_robot import BaseRobot as Robot
# from Robots.sensing_robot import SensingRobot as Robot
# from Robots.smarter_sensing_robot import SmarterSensingRobot as Robot
# from Robots.can_following_robot import CanFollowingRobot as Robot
# from Robots.experimental_robot import ExperimentalRobot as Robot
# from Robots.lookup_table_robot import LookupTableRobot as Robot
# from Reinforcement_Learning.train_q_learner import get_trained_q_learning_robot
# from Reinforcement_Learning.train_q_learner import get_trained_q_learning_robot_optimized

# Choose which robot to test:
# Be sure to comment/uncomment the appropriate import statements above
ROBOT = Robot('Robot under test')
# Q-Learning robots
# ROBOT = get_trained_q_learning_robot('Robot under test')
# ROBOT = get_trained_q_learning_robot_optimized('Robot under test')

NUMBER_OF_ACTIONS = 200
REDRAW_DELAY = 0.2
ENVIRONMENT = Environment()


init(autoreset=True)
env_colour_map = {
    0: Back.BLACK + '  ',
    1: Back.YELLOW + '  ',
    3: Back.BLACK + '🤖',
    4: Back.YELLOW + '🤖',
}

reward_colour_map = {
    -5: Back.RED + '  ',
    -1: Back.MAGENTA + '  ',
    0: Back.BLACK + ' ',
    10: Back.GREEN + ' ',
}

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def display_environment(clear=True):
    if clear:
        clear_screen()
    for row in ENVIRONMENT.printable_grid():
        print(''.join(env_colour_map[val] for val in row))


def display(step, action):
    display_environment()
    print(f'Cans: {ENVIRONMENT.number_of_cans()}')
    print(f'Score: {ROBOT.score}')
    print(f'Reward: {ENVIRONMENT.reward}')
    print(f'Rewards: {''.join(reward_colour_map[val] for val in ENVIRONMENT.recent_rewards(40))}')
    print(f'Time step: {step}')
    print(action)


ENVIRONMENT.randomise(time.time())
ROBOT.set_environment(ENVIRONMENT)
ENVIRONMENT.set_robot(ROBOT)
display_environment()

for i in range(NUMBER_OF_ACTIONS):
    action = ROBOT.choose_action()
    ENVIRONMENT.perform_action(action)
    display(i, action)
    time.sleep(REDRAW_DELAY)
    if ENVIRONMENT.number_of_cans() == 0:
        break
