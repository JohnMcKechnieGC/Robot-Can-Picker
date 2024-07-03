"""
Use Optuna to optimize the hyperparameters for training the QLearningRobot.
"""

from random import seed
import statistics
import optuna
from Robots.q_learning_robot import QLearningRobot
from Problem_Domain.environment import Environment


NUMBER_OF_ACTIONS = 200
NUMBER_OF_TRIALS = 200
NUMBER_OF_ENVIRONMENTS_FOR_EVALUATION = 100


def objective(trial):
    """
    Specify the hyperparameters to tune and appropriate ranges. Then set up a 'study'
    to optimize the internal train function.
    """
    params = {
        'epsilon': trial.suggest_float('epsilon', 0.0, 1.0),
        'decay_factor': trial.suggest_float('decay_factor', 0.0, 1.0),
        'learning_rate': trial.suggest_float('learning_rate', 0.0, 1.0),
        'discount_factor': trial.suggest_float('discount_factor', 0.0, 1.0),
        'min_epsilon': trial.suggest_float('min_epsilon', 0.0, 1.0), #0.0006),
        'number_of_episodes': trial.suggest_int('number_of_episodes', 300, 1000)
    }

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

        # Teturn the average of the last NUMBER_OF_ENVIRONMENTS_FOR_EVALUATION episodes.
        # This helps us to avoid the initial environments when the robot was untrained.
        return statistics.mean(scores[-NUMBER_OF_ENVIRONMENTS_FOR_EVALUATION:])

    score = train(**params)
    return score


# Create a study object that will find the hyperparameters that minimize the objective
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=NUMBER_OF_TRIALS)

# Fetch the best parameters and the best score achieved
best_params = study.best_params
best_score = study.best_value

print("Best score:", best_score)
print("Best parameters:", best_params)
