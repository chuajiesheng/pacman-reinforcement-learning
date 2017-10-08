# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random, util, math
from collections import deque
import tensorflow as tf
import numpy as np


class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """

    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        self.q = util.Counter()
        "*** YOUR CODE HERE ***"

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        return self.q[(state, action)]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        if len(self.getLegalActions(state)) == 0:
            return 0.0

        q_values = [self.getQValue(state, action) for action in self.getLegalActions(state)]
        return max(q_values)

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        max_action = None
        max_q = 0
        for action in self.getLegalActions(state):
            q = self.getQValue(state, action)
            if q > max_q or max_action is None:
                max_q = q
                max_action = action
        return max_action

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        "*** YOUR CODE HERE ***"
        if util.flipCoin(self.epsilon):
            legal_actions = self.getLegalActions(state)
            return random.choice(legal_actions)
        else:
            return self.computeActionFromQValues(state)

    def update(self, state, action, next_state, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        if len(self.getLegalActions(next_state)) == 0:
            utility = reward
        else:
            possible_future_reward = [self.getQValue(next_state, next_action) for next_action in self.getLegalActions(next_state)]
            utility = reward + (self.discount * max(possible_future_reward))
        self.q[(state, action)] = (1 - self.alpha) * self.getQValue(state, action) + self.alpha * utility

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self, state)
        self.doAction(state, action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """

    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        q = 0
        for f, v in self.featExtractor.getFeatures(state, action).items():  # use .iteritems() for python2
            q += v * self.weights[f]

        return q

    def update(self, state, action, next_state, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        for f, v in self.featExtractor.getFeatures(state, action).items():  # use .iteritems() for python2
            if len(self.getLegalActions(next_state)) == 0:
                difference = reward - self.getQValue(state, action)
            else:
                look_ahead = [self.getQValue(next_state, nextAction) for nextAction in self.getLegalActions(next_state)]
                difference = (reward + self.discount * max(look_ahead)) - self.getQValue(state, action)

            self.weights[f] = self.weights[f] + self.alpha * difference * v

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass


class TensorFlowQAgent(PacmanQAgent):

    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

        INPUT_NEURONS = 4
        HIDDEN_NEURONS = 64 * 11 * 10
        OUTPUT_NEURONS = 5  # action space
        self.DIRECTIONS = [Directions.NORTH,
                           Directions.SOUTH,
                           Directions.EAST,
                           Directions.WEST,
                           Directions.STOP]

        self.ip = tf.placeholder(tf.float32, [None, INPUT_NEURONS], name="ip")
        self.op = tf.placeholder(tf.float32, [None, OUTPUT_NEURONS], name="op")
        self.hiddenLayer = {'weights': tf.Variable(tf.random_normal([INPUT_NEURONS, HIDDEN_NEURONS])),
                            'biases': tf.Variable(tf.random_normal([HIDDEN_NEURONS]))}
        self.outputLayer = {'weights': tf.Variable(tf.random_normal([HIDDEN_NEURONS, OUTPUT_NEURONS])),
                            'biases': tf.Variable(tf.random_normal([OUTPUT_NEURONS]))}
        self.hiddenLayerOutput = tf.add(tf.matmul(self.ip, self.hiddenLayer['weights']), self.hiddenLayer['biases'])
        self.hiddenLayerOutput = tf.nn.tanh(self.hiddenLayerOutput)
        self.output = tf.add(tf.matmul(self.hiddenLayerOutput, self.outputLayer['weights']), self.outputLayer['biases'])
        self.loss = tf.reduce_mean(tf.square(self.output - self.op))
        self.loss = tf.clip_by_value(self.loss, 1e-10, 1e+20)
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(self.loss)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        replay_memory_size = 20000
        self.replay_memory = deque([], maxlen=replay_memory_size)

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        q = 0
        for f, v in self.featExtractor.getFeatures(state, action).items():  # use .iteritems() for python2
            q += v * self.weights[f]

        return q

    def sample_memories(self, batch_size):
        indices = np.random.permutation(len(self.replay_memory))[:batch_size]
        samples = []
        for idx in indices:
            memory = self.replay_memory[idx]
            samples.append(memory)
        return samples

    def update(self, state, action, next_state, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"

        # Add the experience to replay memory
        value_for_state = []
        value = [v for _, v in self.featExtractor.getFeatures(state, action).items()]
        if len(value) == 3:
            value_for_state.append([0] + value)
        else:
            value_for_state.append(value)

        value_for_next_state = []
        for nextAction in self.getLegalActions(next_state):
            features = [v for _, v in self.featExtractor.getFeatures(next_state, nextAction).items()]
            if len(features) == 3:
                features = [0] + features
            value_for_next_state.append(features)
        done = len(self.getLegalActions(next_state)) == 0

        self.replay_memory.append((value_for_state, action, reward, value_for_next_state, done))
        # Sample sampleBatch, a batch of training data from the replay memory
        batch_size = 50
        sampleBatch = self.sample_memories(batch_size)

        trainingInputStates = []
        trainingTargetValues = []
        for train_state, train_action, train_reward, train_nextState, isTerminal in sampleBatch:
            QValues = self.sess.run([self.output], feed_dict={self.ip: np.array(train_state)})[0][0]
            if isTerminal:
                train_nextState = [[0, 0, 0, 0]]
            nextStateQValues = self.sess.run([self.output], feed_dict={self.ip: np.array(train_nextState)})[0][0]
            maxQVal = max(nextStateQValues)

            # Update rule
            if isTerminal:
                newQVal = train_reward
            else:
                newQVal = (train_reward + self.discount * maxQVal)

            targetQValues = QValues.copy()
            targetQValues[self.DIRECTIONS.index(train_action)] = newQVal
            trainingInputStates.append(train_state[0])
            trainingTargetValues.append(targetQValues)

        # Optimize
        self.sess.run([self.optimizer], feed_dict={self.ip: trainingInputStates, self.op: trainingTargetValues})

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
