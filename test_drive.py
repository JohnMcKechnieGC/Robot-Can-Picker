"""
Show the progress of the robot under test.
Key: 0 - empty square
     1 - square contains a can
     3 - square contains robot
     4 - square contains robot and a can
"""

from Problem_Domain.environment import Environment
# from Robots.base_robot import BaseRobot as Robot
# from Robots.sensing_robot import SensingRobot as Robot
# from Robots.smarter_sensing_robot import SmarterSensingRobot as Robot
# from Robots.can_following_robot import CanFollowingRobot as Robot
# from Robots.experimental_robot import ExperimentalRobot as Robot
# from Robots.lookup_table_robot import LookupTableRobot as Robot
from Reinforcement_Learning.train_q_learner import get_trained_q_learning_robot


NUMBER_OF_ACTIONS = 200
NUMBER_OF_TRIALS = 1

#test_robot = Robot('Robot under test')
#test_robot = get_trained_q_learning_robot('Robot under test')

# Optimised Q-Learner
test_robot = get_trained_q_learning_robot('Robot under test',
                                          epsilon=0.5516791135810103,
                                          decay_factor=0.6068794434698493,
                                          learning_rate=0.9994968692336417,
                                          discount_factor=0.3559625242060554,
                                          min_epsilon=7.088018732062675e-06,
                                          number_of_episodes=896)
environment = Environment()

total_score = 0

for _ in range(NUMBER_OF_TRIALS):
    environment.randomise()
    test_robot.set_environment(environment)
    environment.set_robot(test_robot)
    environment.display(test_robot.x, test_robot.y)
    for i in range(NUMBER_OF_ACTIONS):
        action = test_robot.choose_action()
        environment.perform_action(action)
        environment.display(test_robot.x, test_robot.y)
        print(action, test_robot.score, i)
    print(test_robot.score)
    total_score += test_robot.score

print('Average score:', total_score / NUMBER_OF_TRIALS)
