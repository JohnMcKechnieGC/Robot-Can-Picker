"""
Q-Learning Pseudocode:

Initialize Q(s, a) arbitrarily, for all s, a
For each episode:
   Initialize state s as the starting position (typically the top-left corner)
   For each step in the episode (defualt maximum of 200 steps/actions per episode):
       Choose action a from state s using policy derived from Q (e.g., ε-greedy)
       Take action a, observe reward r, and next state s'
       Update Q-value for the state-action pair (s, a) using the formula:
           Q(s, a) <- Q(s, a) + α * (r + γ * max(Q(s', a')) - Q(s, a))
           where α is the learning rate and γ is the discount factor.
       s <- s' // Move to the new state
       If s is terminal, break out of the loop
   End For
End For
"""

from random import seed
from Problem_Domain.environment import Environment
from Robots.q_learning_robot import QLearningRobot


NUMBER_OF_ACTIONS = 200

def get_trained_q_learning_robot(name,
                                 number_of_episodes=400,
                                 epsilon=0.99,
                                 decay_factor=0.99,
                                 learning_rate=0.1,
                                 discount_factor=0.9,
                                 min_epsilon=0.1):
    """
    Implement the Q-Learning algorithm above with respect to our QLearningRobot.
    """
    seed(42)
    learning_robot = QLearningRobot(name,
                                    epsilon=epsilon,
                                    decay_factor=decay_factor,
                                    learning_rate=learning_rate,
                                    discount_factor=discount_factor,
                                    min_epsilon=min_epsilon)
    environment = Environment()

    for _ in range(number_of_episodes):
        environment.randomise()
        learning_robot.set_environment(environment)
        environment.set_robot(learning_robot)
        for _ in range(NUMBER_OF_ACTIONS):
            action = learning_robot.choose_action(is_learning=True)
            environment.perform_action(action)
            learning_robot.reinforce(environment.reward)
        learning_robot.decay_epsilon()
    learning_robot.score = 0
    return learning_robot


def get_trained_q_learning_robot_optimized(name):
    """
    Get a QLearningRobot using hyperparameters optimized by Optuna. 
    """
    # return get_trained_q_learning_robot(name,
    #                                     epsilon=0.5516791135810103,
    #                                     decay_factor=0.6068794434698493,
    #                                     learning_rate=0.9994968692336417,
    #                                     discount_factor=0.3559625242060554,
    #                                     min_epsilon=7.088018732062675e-06,
    #                                     number_of_episodes=896)
    return get_trained_q_learning_robot(name,
                                        epsilon=0.6,
                                        decay_factor=0.3,
                                        learning_rate=0.9,
                                        discount_factor=0.5,
                                        min_epsilon=0.0,
                                        number_of_episodes=1200)

# 'learning_rate': 0.9, 'discount_factor': 0.5, 'min_epsilon': 0.0, 'number_of_episodes': 1200}
