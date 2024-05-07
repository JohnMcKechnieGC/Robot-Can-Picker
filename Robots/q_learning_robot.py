from Robots.base_robot import BaseRobot


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
    pass
