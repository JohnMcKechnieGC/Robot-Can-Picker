"""
Compare all of our robots across a consistent set of randomly generated environments. 
"""

from random import randint
from Problem_Domain.environment import Environment
from Robots.base_robot import BaseRobot
from Robots.sensing_robot import SensingRobot
from Robots.smarter_sensing_robot import SmarterSensingRobot
from Robots.can_following_robot import CanFollowingRobot
from Robots.experimental_robot import ExperimentalRobot
from Robots.lookup_table_robot import LookupTableRobot
from Reinforcement_Learning.train_q_learner import get_trained_q_learning_robot


NUMBER_OF_ACTIONS = 200
NUMBER_OF_TRIALS = 1000

# Create a list of random seeds, one for each trial, to ensure that each robot gets
# the same set of randomly generated environments.
RANDOM_SEEDS = [randint(-2147483648, 2147483647) for _ in range(NUMBER_OF_TRIALS)]


def evaluate_robot(robot):
    """
    Evaluate a single robot and return the average score across all trials.
    """
    total_score = 0
    for seed in RANDOM_SEEDS:
        environment = Environment(random_seed=seed)
        robot.set_environment(environment)
        environment.set_robot(robot)
        for _ in range(NUMBER_OF_ACTIONS):
            environment.perform_action(robot.choose_action())
        total_score += robot.score
    print(f'Average score for {robot.name}: {total_score / NUMBER_OF_TRIALS}')


def evaluate_all_robots(robots_under_test):
    """
    Evaluate each robot in turn.
    """
    for robot in robots_under_test:
        evaluate_robot(robot)


if __name__ == '__main__':
    robots = [BaseRobot('Random Robbie'),
              SensingRobot('Sensing Sadie'),
              SmarterSensingRobot('Smarter Sadie'),
              CanFollowingRobot('Can Magnet'),
              ExperimentalRobot('Experimental'),
              LookupTableRobot('Evolved'),
              get_trained_q_learning_robot('Default Q-Learner'),
              get_trained_q_learning_robot(
                  'Optimised Q-Learner',
                  epsilon=0.5516791135810103,
                  decay_factor=0.6068794434698493,
                  learning_rate=0.9994968692336417,
                  discount_factor=0.3559625242060554,
                  min_epsilon=7.088018732062675e-06,
                  number_of_episodes=896)]
    evaluate_all_robots(robots)
