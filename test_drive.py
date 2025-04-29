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
from Robots.can_following_robot import CanFollowingRobot as Robot
# from Robots.experimental_robot import ExperimentalRobot as Robot
# from Robots.lookup_table_robot import LookupTableRobot as Robot
# from Reinforcement_Learning.train_q_learner import get_trained_q_learning_robot
# from Reinforcement_Learning.train_q_learner import get_trained_q_learning_robot_optimized


NUMBER_OF_TRIALS = 1
NUMBER_OF_ACTIONS = 200

# Choose which robot to test:
# Be sure to comment/uncomment the appropriate import statements above
test_robot = Robot('Robot under test')
# Q-Learning robots
# test_robot = get_trained_q_learning_robot_optimized('Robot under test')
# test_robot = get_trained_q_learning_robot('Robot under test')

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
