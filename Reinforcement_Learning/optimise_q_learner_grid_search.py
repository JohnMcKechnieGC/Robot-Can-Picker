"""
Find the best parameters for the QLearningRobot using a grid search.
Warning: This script will take a long time to run as it's currently configured.
"""
from random import seed
import itertools
import statistics
from Robots.q_learning_robot import QLearningRobot
from Problem_Domain.environment import Environment

NUMBER_OF_ACTIONS = 200
NUMBER_OF_ENVIRONMENTS_FOR_EVALUATION = 100


def train(epsilon,
          decay_factor,
          learning_rate,
          discount_factor,
          min_epsilon,
          number_of_episodes):
    """
    Train the QLearningRobot using the specified parameters. Then take the average score for the
    last NUMBER_OF_TRIALS_FOR_EVALUATION and use that as the value to optimize.
    """
    # Set the random seed to keep the comparison of different robots fair, i.e., the same
    # sequence of randomly generated environments will be generated.
    seed(42)
    learning_robot = QLearningRobot('Q-Learner',
                                    epsilon=epsilon,
                                    decay_factor=decay_factor,
                                    learning_rate=learning_rate,
                                    discount_factor=discount_factor,
                                    min_epsilon=min_epsilon)
    environment = Environment()
    scores = []
    for _ in range(number_of_episodes):
        environment.randomise()
        learning_robot.set_environment(environment)
        environment.set_robot(learning_robot)
        for _ in range(NUMBER_OF_ACTIONS):
            action = learning_robot.choose_action(is_learning=True)
            environment.perform_action(action)
            learning_robot.reinforce(environment.reward)
        learning_robot.decay_epsilon()
        scores.append(learning_robot.score)

    # Return the average of the last NUMBER_OF_ENVIRONMENTS_FOR_EVALUATION episodes.
    # This helps us to avoid the initial environments when the robot was untrained.
    return statistics.mean(scores[-NUMBER_OF_ENVIRONMENTS_FOR_EVALUATION:])


def perform_grid_search(parameters):
    best_score = float('inf') * -1
    keys = list(parameters.keys())
    # itertools.product generates the Cartesian product of the provided
    # parameter lists, meaning that every possible combination is generated.
    for values in itertools.product(*(parameters[key] for key in keys)):
        # The zip function pairs each key with its corresponding value
        # from the current permutation.
        permutation = dict(zip(keys, values))
        score = train(**permutation)
        if score > best_score:
            best_score = score
            print(best_score, permutation)


q_learning_params = {
    'epsilon': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'decay_factor': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'learning_rate': [0.01, 0.1, 0.5, 0.9, 1.0],
    'discount_factor': [0.1, 0.2, 0.3, 0.4, 0.5],
    'min_epsilon': [0.1, 0.01, 0.0],
    'number_of_episodes': [400, 900, 1200]
    }


perform_grid_search(q_learning_params)
