import gym
import numpy as np
import sys
from collections import deque
# sys.path.append('/usr/local/lib/python2.7/site-packages')
import cv2

from keras.models import load_model, Sequential
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam
from keras.layers.core import Activation, Dropout, Flatten, Dense

# Preprocess
# # DOWNLOAD CV2
def preprocess(observation):
    observation = cv2.cvtColor(cv2.resize(observation, (84, 110)), cv2.COLOR_BGR2GRAY)
    observation = observation[26:110,:]
    ret, observation = cv2.threshold(observation,1,255,cv2.THRESH_BINARY)
    return np.reshape(observation,(84,84,1))



# Load game's environment
env = gym.make('SpaceInvaders-v0')
# Check game's specs
actions = env.action_space.n
# print(actions)
# print(env.observation_space.shape)


# Q-Learning init
# Q = np.zeros([84, 84, actions])
# eta = .628
epsilon = 1.
gamma = .9
epochs = 5000
memory = deque()

# Initialize neural network
model = Sequential()
model.add(Dense(128, input_shape=(84,84), activation='relu'))
# model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(actions, activation='linear'))
model.compile(loss='mse', optimizer=Adam())
print("Successfully constructed networks.")

# Predict action
def act(state):
    if np.random.rand() <= epsilon:
        return random.randrange(actions)
    act_values = model.predict(state)
    return np.argmax(act_values[0])  # returns action


# Train model
for e in range(10):

    state = env.reset()
    state = preprocess(state)

    for play in range(100):
        # Get action
        action = act(state)
        next_state, reward, done, _ =env.step(action)
        next_state = preprocess(next_state)

        memory.append((state, action, reward, next_state, done))

        state = next_state
        if done:
            print('Done at play: ', play)
            break


minibatch = random.sample(memory, batch_size)
for state, action, reward, next_state, done in minibatch:
    target = reward
    if not done:
      target = reward + gamma * \
               np.amax(model.predict(next_state)[0])
    target_f = model.predict(state)
    target_f[0][action] = target
    model.fit(state, target_f, epochs=1, verbose=0)
if epsilon > epsilon_min:
    epsilon *= epsilon_decay
