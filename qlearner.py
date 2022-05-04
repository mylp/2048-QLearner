import random
import numpy as np
from game import *
import curses
import utils

class QLearner:
    def __init__(self):
        self.qvs = {}
        self.alpha = 0.9
        self.epsilon = 0.01
        self.discount = 0.8

    def getLegalActions(self, state):
        return 'Up' # Test purposes

    def getQValue(self, state, action):
        """
          Returns Q(state,action). Unseen states are 0.0
        """
        if (state, action) in self.qvs:
          return self.qValues[(state, action)]
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
        if util.flipCoin(self.epsilon):
          return random.choice(legalActions)
        else:
          return self.getPolicy(state)

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
        """
        q = self.getQValue(state, action)
        value = self.getValue(nextState)
        new_q = (1-self.alpha) * q + self.alpha * (reward + self.discount*value)
        self.qValues[(state, action)] = new_q

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)

def __main__():
  pass