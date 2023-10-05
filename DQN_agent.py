import numpy as np
import random
from collections import deque
from DQN_model import DQN

class DQNAgent:
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.memory = deque(maxlen=1000)
        self.gamma = 0.95  # factor de descuento
        self.epsilon = 1.0  # exploración inicial
        self.epsilon_decay = 0.995  # reducción de la exploración
        self.epsilon_min = 0.1  # exploración mínima
        self.model = DQN(num_actions)
        self.target_model = DQN(num_actions)
        self.update_target_model()

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.num_actions)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.target_model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
