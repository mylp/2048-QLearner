import random
import numpy as np
from game import *
import curses
import util

actions = ['Up', 'Left', 'Down', 'Right', 'Restart', 'Exit']


class QLearner:
    def __init__(self):
        self.field = GameField(win=2**10)
        self.qvs = util.Counter()
        self.alpha = 0.9
        self.epsilon = 0.01
        self.discount = 0.8
        self.episodeRewards = 0

    def getLegalActions(self, state):
        legals = []
        for action in actions:
            if state.move_is_possible(action):
                legals.append(action)
        return legals

    def getQValue(self, state, action):
        """
          Returns Q(state,action). Unseen states are 0.0
        """
        if (state, action) in self.qvs:
            return self.qvs[(state, action)]
        else:
            return 0.0

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.
        """
        legal = self.getLegalActions(state)
        if not len(legal):
            return 0.0
        else:
            return max([self.getQValue(state, action) for action in legal])

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.
        """
        legal = self.getLegalActions(state)
        if not len(legal):
            return None
        actions = util.Counter()
        for action in legal:
            actions[action] = self.getQValue(state, action)
        return actions.argMax()

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.
        """
        legalActions = self.getLegalActions(state)
        if not len(legalActions):
            return None
        if random.random() < self.epsilon:
            return random.choice(legalActions)
        else:
            return self.computeActionFromQValues(state)

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
        """
        q = self.getQValue(state, action)
        value = self.computeValueFromQValues(state)
        new_q = (1-self.alpha) * q + self.alpha * \
            (reward + self.discount*value)
        self.qvs[(state, action)] = new_q

    def observe(self, state, action, nextState, deltaReward):
        """
            Called by environment to inform agent that a transition has
            been observed.
        """
        self.episodeRewards += deltaReward
        self.update(state, action, nextState, deltaReward)

    def teach(self):
        field = self.field
        playing = True

        def learning():
            action = self.getAction(field)
            old_field = deepcopy(field)
            if field.move(action):
                field.moveCount += 1
                # Need to tweak reward system
                self.observe(old_field, action, field,
                             (field.score - old_field.score))
                if field.is_gameover():
                    return False
            return True

        while playing:
            playing = learning()


def __main__():
    agent, agent2, highscore = QLearner(), QLearner(), 0

    for _ in range(50):
        agent.teach()
        highscore = max(highscore, agent.field.highscore)
        agent.field.reset()
    print("Best highscore after 50 episodes: ", highscore)

    highscore = 0

    for _ in range(1000):
        agent2.teach()
        highscore = max(highscore, agent2.field.highscore)
        agent2.field.reset()
    print("Best highscore after 1000 episodes: ", highscore)


__main__()
