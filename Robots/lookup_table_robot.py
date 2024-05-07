from Robots.base_robot import BaseRobot


class LookupTableRobot(BaseRobot):
    def __init__(self, name='', action_lookup_table=None):
        super().__init__(name)
        if action_lookup_table is None:
            # This default lookup table is the result of evolving the robot controller using a genetic algorithm.
            action_lookup_table =\
                [2, 2, 5, 4, 4, 4, 5, 5, 5, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 6, 6, 6, 6, 2, 2, 6, 4, 5, 5, 5, 5, 5, 1, 3,
                 4, 5, 1, 1, 1, 5, 1, 6, 6, 1, 1, 6, 6, 2, 1, 4, 4, 2, 2, 3, 3, 6, 3, 4, 2, 2, 5, 7, 5, 1, 3, 1, 1, 4,
                 7, 4, 2, 2, 6, 4, 4, 3, 2, 3, 2, 4, 4, 3, 5, 5, 5, 2, 6, 5, 3, 5, 3, 3, 1, 1, 1, 3, 1, 1, 1, 4, 6, 2,
                 7, 3, 1, 6, 4, 5, 2, 2, 5, 5, 1, 5, 3, 1, 5, 5, 1, 1, 3, 5, 6, 3, 3, 1, 3, 2, 2, 5, 4, 6, 7, 4, 2, 3,
                 6, 3, 3, 6, 6, 6, 5, 6, 1, 6, 1, 1, 1, 2, 1, 6, 6, 2, 1, 3, 7, 6, 2, 2, 6, 6, 4, 2, 5, 4, 2, 1, 2, 2,
                 4, 1, 1, 6, 1, 1, 4, 1, 1, 5, 1, 4, 4, 4, 4, 1, 4, 4, 2, 4, 1, 3, 5, 2, 1, 1, 2, 7, 7, 5, 1, 2, 1, 5,
                 1, 7, 3, 4, 2, 4, 2, 3, 4, 3, 2, 1, 4, 7, 5, 4, 1, 5, 5, 2, 6, 1, 2, 6, 1, 7, 7, 5, 7, 4, 7, 7, 1, 3,
                 4, 2, 2, 6, 1]
        self.action_lookup_table = action_lookup_table

    def set_lookup_table(self, action_lookup_table):
        self.action_lookup_table = action_lookup_table

    def choose_action(self):
        self.sense_environment()
        self.calculate_situation_number()
        action_number = self.action_lookup_table[self.situation_number] - 1  # - 1 because actions are zero based
        return self.actions[action_number]
