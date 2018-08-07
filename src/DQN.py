import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, MaxPooling2D, LSTM
from keras.optimizers import Adam
from collections import deque
import random


# Deep Q-learning Agent
class DQN:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma=0.95
        self.epsilon=1.0
        self.epsilon_min=0.01
        self.epsilon_decay=0.995
        self.learning_rate=0.001

        self.batch_size=32
        self.episodes=1000

        self.memory = deque(maxlen=2000)

        self.model = self._build_model_linear()


    """ From Mnih et al. "Playing atari with deep reinforcement learning." 2013. """
    def _build_model(self):
        model = Sequential()

        # Convolution Layers
        model.add(Convolution2D(32,(8,8), strides=(4,4), activation='relu',
                                input_shape=self.state_size))

        model.add(Convolution2D(64,(4,4), strides=(2,2), activation='relu'))
        model.add(Convolution2D(64,(3,3), activation='relu'))

        # Fully Connected Layers
        model.add(Flatten())
        model.add(Dense(512, activation='relu', input_dim=self.state_size))
        model.add(Dense(self.action_size, activation='linear'))

        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        return model

    def _build_model_linear(self):
        model = Sequential()
        model.add(Dense(24, activation='relu', input_shape=self.state_size))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def load(file, training):
        # Load model from file
        self.model = load_model('drqn_model.h5')
        
        if not training:
            # Only want greedy decisions
            self.epsilon = 0.0

    """ Save values to memory """
    def remember(self, state, action, reward, nextState, done):
        self.memory.append((state, action, reward, nextState, done))

    """ Carry out best or random action """
    def act(self, state):
        #print state
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        action_values = self.model.predict(state)
        return np.argmax(action_values[0])

    """ Replay a minibatches """
    def replay(self, batch_size):
        batch_size = min(self.batch_size, len(self.memory))
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = 0
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def get_model():
        return self.model

""" Deep Recurrent Q-Learning Network """
""" https://arxiv.org/pdf/1507.06527.pdf """
class DRQN(DQN):
    def __init__(self, state_size, action_size):
        super().__init__(state_size, action_size)
        self.model = self._build_model()

    def _build_model(self):
        # Input = [pedx-x, pedy-y, orthogonalVelocitiesDifference, pedTheta-Theta, headTheta]^T
        model = Sequential()

        # Convolution Layers
        model.add(Convolution2D(32,(8,8), strides=(4,4), activation='relu',
                                input_shape=self.state_size))

        model.add(Convolution2D(64,(4,4), strides=(2,2), activation='relu'))
        model.add(Convolution2D(64,(3,3), activation='relu'))

        # Fully Connected Layers
        model.add(Flatten())
        model.add(LSTM(units=32))

        model.add(Dense(self.action_size, activation='linear'))

        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        return model
