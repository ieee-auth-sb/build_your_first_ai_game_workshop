import gym
import numpy as np
import sys
from collections import deque
import cv2, random
import matplotlib.pyplot as plt

from keras.models import load_model, Sequential
from keras.layers.convolutional import Conv2D
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

#Testing Preprocess Photo
env.reset()
action0 = 0  # do nothing
observation0, reward0, terminal, info = env.step(action0)
print("Before processing: " + str(np.array(observation0).shape))
plt.imshow(np.array(observation0))
plt.show()
observation0 = preprocess(observation0)
print("After processing: " + str(np.array(observation0).shape))
plt.imshow(np.array(np.squeeze(observation0)))
plt.show()

# # SET CORRECT EPSILONS
epsilon = 1.
epsilon_min = 0.1
epsilon_decay = 0.1
gamma = .9
batch_size = 4
episodeNumber = 100
memory = deque()
test = False

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
for episode in range(episodeNumber):
    state = env.reset()
    state = preprocess(state)
    env.render()
    for play in range(10000):
        # Get action
        action = act(state)
        next_state, reward, done, _ = env.step(action)
        next_state = preprocess(next_state)
        if (len(memory) > 10000):
            memory.popleft()
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
    print("Episode: ", episode)
    for state, action, reward, next_state, done in minibatch:
        target = reward
        # Update target
        if not done:
            target = reward + gamma * \
                np.amax(model.predict(next_state, batch_size=1))
        target_f = model.predict(state, batch_size=1)
        target_f[0][action] = target
        model.fit(state, target_f, epochs=1, verbose=2)
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay


# SAVE MODEL
model.save("model.h5")
