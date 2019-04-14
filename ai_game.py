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
def preprocess(observation):
    observation = cv2.cvtColor(cv2.resize(observation, (84, 110)), cv2.COLOR_BGR2GRAY)
    observation = observation[26:110,:]
    ret, observation = cv2.threshold(observation,1,255,cv2.THRESH_BINARY)
    return np.reshape(observation,(1,84,84,1))



# Load game's environment
env = gym.make('SpaceInvaders-v0')
# Check game's specs
actions = env.action_space.n
# print(actions)
# print(env.observation_space.shape)

# # SET CORRECT EPSILONS
epsilon = 1.
epsilon_min = 0.1
epsilon_decay = 0.1
gamma = .9
epochs = 5000
batch_size = 4
memory = deque()

# Initialize neural network
model = Sequential()
model.add(Conv2D(32, (8, 8), strides=(
    4, 4), input_shape=(84, 84, 1), activation='relu'))
model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense((512), activation='relu'))
model.add(Dense(actions))
model.compile(loss='mse', optimizer=Adam())
model.summary()
print("Model constructed")

# Predict action
def act(state):
    if np.random.rand() <= epsilon:
        return random.randrange(actions)
    act_values = model.predict(state)
    return np.argmax(act_values[0])  # returns action

# Collect data
for e in range(10):

    state = env.reset()
    state = preprocess(state)

    for play in range(1000):
        # Get action
        action = act(state)
        next_state, reward, done, _ = env.step(action)
        next_state = preprocess(next_state)
        memory.append((state, action, reward, next_state, done))
        env.render()
        state = next_state
        if done:
            print('Done at play: ', play)
            break
    if not done:
        print('Play stopped at: ', play)

    # Train model
    # Sample plays
    minibatch = random.sample(memory, batch_size)
    for state, action, reward, next_state, done in minibatch:
        target = reward
        # Update target
        if not done:
            target = reward + gamma * \
                np.amax(model.predict(next_state, batch_size=1))
        target_f = model.predict(state, batch_size=1)
        target_f[0][action] = target
        model.fit(state, target_f, epochs=1, verbose=0)
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay


# TO DO: SAVE MODEL AFTER SOME TIME
