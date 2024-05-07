from Robots.base_robot import BaseRobot
from random import random, randint


class QLearningRobot(BaseRobot):
    # Initialize Q(s, a) arbitrarily, for all s, a
    # For each episode:
    #    Initialize state s as the starting position(typically the top-left corner)
    #    For each step in the episode(maximum of 200 steps per episode):
    #        Choose action a from state s using policy derived from Q(e.g., ε-greedy)
    #        Take action a, observe reward r, and next state s'
    #        Update Q-value for the state-action pair(s, a) using the formula:
    #            Q(s, a) < - Q(s, a) + α * (r + γ * max(Q(s', a')) - Q(s, a))
    #        s < - s' // Move to the new state
    #        If s is terminal, break out of the loop
    #    End For
    # End For
    def __init__(self, name='', epsilon=0.99, decay_factor=0.99, learning_rate=0.1, discount_factor=0.9):
        super().__init__(name)
        self.epsilon = epsilon
        self.decay_factor = decay_factor
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.action_number = -1
        # Initialize Q-table:
        # Each state/action pair is assigned a small random value
        MAX_STATES = 3 ** 5
        self.q_table = [[random(), random(), random(), random(), random(), random(), random()]
                        for _ in range(MAX_STATES)]
    
    def decay_epsilon(self):
        self.epsilon *= self.decay_factor

    def is_choosing_random_action(self):
        # Decide to choose a random action if a random value [0..1) <= learning rate
        return random() <= self.epsilon

    def choose_action(self, is_learning=True):
        self.sense_environment()
        # If the robot is still exploring the range of possible actions and rather than
        # just exploiting what it has learned, and if a random action should be chosen
        if is_learning and self.is_choosing_random_action():
            self.action_number = randint(0, len(self.actions) - 1)  # pick a random action
        else:
            # Pick the best known action
            self.calculate_situation_number()
            q_values = self.q_table[self.situation_number]
            max_q = max(q_values)
            self.action_number = q_values.index(max_q)

        # Return the chosen action
        return self.actions[self.action_number]

    def reinforce(self, reward):
        # Q(s, a) < - Q(s, a) + α * (r + γ * max(Q(s', a')) - Q(s, a))
        # Calculate max(Q(s', a'))
        previous_situation_number = self.situation_number
        self.sense_environment()
        self.calculate_situation_number()
        max_next_q = max(self.q_table[self.situation_number])

        current_q_value = self.q_table[previous_situation_number][self.action_number]
        current_q_value += \
            self.learning_rate * (reward + self.discount_factor * max_next_q - current_q_value)
        self.q_table[previous_situation_number][self.action_number] = current_q_value
