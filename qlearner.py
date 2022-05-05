import random
import numpy as np
from game import *
import curses
import util

actions = ['Up', 'Left', 'Down', 'Right', 'Restart', 'Exit']

class QLearner:
    def __init__(self):
        self.field = GameField(win=2**17)
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
          return self.getPolicy(state)

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
        """
        q = self.getQValue(state, action)
        value = self.getValue(nextState)
        new_q = (1-self.alpha) * q + self.alpha * (reward + self.discount*value)
        self.qvs[(state, action)] = new_q

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)

    def observeTransition(self, state,action,nextState,deltaReward):
        """
            Called by environment to inform agent that a transition has
            been observed.
        """
        self.episodeRewards += deltaReward
        self.update(state,action,nextState,deltaReward)

    def play(self):
        state_actions = {}
        field = self.field

        def init():
            field.reset()
            return 'Game'

        state_actions['Init'] = init

        def not_game(state):
            action = 'Exit'
            responses = defaultdict(lambda: state)
            responses['Restart'], responses['Exit'] = 'Init', 'Exit'
            return responses[action]
        
        state_actions['Win'] = lambda: not_game('Win')
        state_actions['Gameover'] = lambda: not_game('Gameover')

        def game():
            action = self.getAction(field)
            if action == 'Restart':
                return 'Init'
            if action == 'Exit':
                return 'Exit'
            temp_field = deepcopy(field)
            current_score = temp_field.score
            if field.move(action):
                score_diff = field.score - current_score
                self.observeTransition(field, action, temp_field, score_diff)
                if field.is_win():
                    return 'Win'
                if field.is_gameover():
                    return 'Gameover'
            return 'Game'
            
        state_actions['Game'] = game
        state = 'Init'
        while state != 'Exit':
            state = state_actions[state]()

def __main__():

  """ 
  How do we know if the agent is learning?

  """


  agent = QLearner()
  sum_highscore = 0
  for i in range(50):
    agent.play()
    agent.field.reset()
    sum_highscore += agent.field.highscore
  print("Average highscore after 50 iterations: ", sum_highscore/50)

  agent2 = QLearner()
  sum_highscore = 0
  for i in range(200):
    agent2.play()
    agent2.field.reset()
    sum_highscore += agent2.field.highscore
  print("Average highscore after 200 iterations: ", sum_highscore/200)

__main__()