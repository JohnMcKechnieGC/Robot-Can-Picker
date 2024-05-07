from environment import Environment
from Robots.q_learning_robot import QLearningRobot as Robot


NUMBER_OF_ACTIONS = 200
NUMBER_OF_TRIALS = 300

test_robot = Robot('Robot under test')
environment = Environment()

for _ in range(NUMBER_OF_TRIALS):
    environment.randomise()
    test_robot.set_environment(environment)
    environment.set_robot(test_robot)
    environment.display(test_robot.x, test_robot.y)
    for i in range(NUMBER_OF_ACTIONS):
        action = test_robot.choose_action()
        environment.perform_action(action)
        test_robot.reinforce(environment.reward)
        #environment.display(test_robot.x, test_robot.y)
        #print(f'{action}, Current: {test_robot.score}, Reward: {environment.reward}')
    environment.display(test_robot.x, test_robot.y)
    test_robot.decay_epsilon()
    print(test_robot.score, test_robot.epsilon)
    test_robot.score = 0
