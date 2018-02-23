import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, MaxPooling2D


# Deep Q-learning Agent Class
class DQN:
    def __init__(self, input_size, output_size, state_size, action_size):
        self.input_size = input_size
        self.output_size = output_size
        self.state_size = state_size
        self.action_size = action_size
        self.gamma=0.95
        self.epsilon=0.9
        self.epsilon_min=0.01
        self.epsilon_decay=0.995
        self.learning_rate=0.001
        
        self.memory = deque(maxlen=2000)

        self.model = self._build_model()
        
    
    """ From Mnih et al. "Playing atari with deep reinforcement learning." 2013. """
    def _build_model(self):
        model = Sequential()

        # Convolution Layers
        model.add(Convolution2D(32,(8,8),(4,4), activation='relu',
                                input_shape=self.input_shape))

        model.add(Convolution2D(64,(4,4),(2,2), activation='relu'))
        model.add(Convolution2D(64,(3,3), activation='relu'))

        # Fully Connected Layers
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))

        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        
        return model

    """ Save values to memory """
    def remember(self, state, action, reward, nextState, done):
        self.memory.append((state, action, reward, nextState, done))
    
    """ Carry out best or random action """
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        action_values = self.model.predict(state)
        return np.argmax(action_values[0])

    """ Replay a minibatches """
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.argmax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
