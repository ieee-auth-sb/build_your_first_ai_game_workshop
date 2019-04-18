# --STEP 0--
# Import and initialize Mountain Car Environment
import gym
import numpy as np
import matplotlib.pyplot as plt

env=gym.make('MountainCar-v0')
env.reset()


# --STEP 1--

# Set the variables
learning = 0.2
discount = 0.9
epsilon = 0.8
min_eps = 0
episodes = 5000

# Determine size of discretized state space

num_states = (env.observation_space.high - env.observation_space.low)*np.array([10, 100])
num_states = np.round(num_states, 0).astype(int) + 1


# --STEP 2--

# Initialize variables to track rewards
reward_list = []
ave_reward_list = []

# Initialize Q table
Q = np.random.uniform(low = -1, high = 1, size = (num_states[0], num_states[1], env.action_space.n))


# --STEP 3--

# Initialize parameters (done,state,rewards)
done = False
tot_reward, reward = 0,0
state = env.reset()


# Discretize state
state_adj = (state - env.observation_space.low)*np.array([10, 100])
state_adj = np.round(state_adj, 0).astype(int)


# --STEP 4--

#Create a loop that is terminated when the game is won
#MAKE SURE THAT YOUR CODE IS ALLIGNED CORRECTLY

while done != True:

    # Render environment
    env.render()


    # Determine next action - epsilon greedy strategy
    if np.random.random() < 1 - epsilon:
           action = np.argmax(Q[state_adj[0], state_adj[1]])
    else:
           action = np.random.randint(0, env.action_space.n)

    # Get next state and reward
    state2, reward, done, info = env.step(action)


    # Discretize new state
    state2_adj = (state2 - env.observation_space.low)*np.array([10, 100])
    state2_adj = np.round(state2_adj, 0).astype(int)


    #Allow for terminal states
    if done and state2[0] >= 0.5:
          Q[state_adj[0], state_adj[1], action] = reward


    # Adjust Q value for current state / Apply the Q-Learning function
    else:
          Q[state_adj[0], state_adj[1], action] = (1-learning) *Q[state_adj[0], state_adj[1], action] +learning * (reward + discount*Q[state2_adj[0], state2_adj[1],np.argmax(Q[state2_adj[0], state2_adj[1]]) ])




    # Update variables
    tot_reward += reward
    state_adj = state2_adj

#Close the environment after the loop
env.close()


# --STEP 5--

# Calculate episodic reduction in epsilon
reduction = (epsilon - min_eps)/1000


#Create the loop for the episodes assigned in the beginning of the code and put inside the code from STEP 3
#AND STEP 4 but now close the environment at the end of for loop and
#Rememer to render the environment every 200 episode this time


#MAKE SURE THAT YOUR CODE IS ALLIGNED CORRECTLY


for i in range(episodes):
    # Initialize parameters (done,state,rewards)
    done = False
    tot_reward, reward = 0,0
    state = env.reset()


    # Discretize state
    state_adj = (state - env.observation_space.low)*np.array([10, 100])
    state_adj = np.round(state_adj, 0).astype(int)




    while done != True:

        # Render environment
        if (i+1)%200 == 0:
            env.render()


        # Determine next action - epsilon greedy strategy
        if np.random.random() < 1 - epsilon:
               action = np.argmax(Q[state_adj[0], state_adj[1]])
        else:
               action = np.random.randint(0, env.action_space.n)

        # Get next state and reward
        state2, reward, done, info = env.step(action)


        # Discretize new state
        state2_adj = (state2 - env.observation_space.low)*np.array([10, 100])
        state2_adj = np.round(state2_adj, 0).astype(int)


        #Allow for terminal states
        if done and state2[0] >= 0.5:
              Q[state_adj[0], state_adj[1], action] = reward


        # Adjust Q value for current state / Apply the Q-Learning function
        else:
              Q[state_adj[0], state_adj[1], action] = (1-learning) *Q[state_adj[0], state_adj[1], action] +learning * (reward + discount*Q[state2_adj[0], state2_adj[1],np.argmax(Q[state2_adj[0], state2_adj[1]]) ])




        # Update variables
        tot_reward += reward
        state_adj = state2_adj

    # Inside the for loop you need to decay epsilon
    if epsilon > min_eps:
        epsilon -= reduction

    # Track rewards
    reward_list.append(tot_reward)

    if (i+1) % 100 == 0:
        ave_reward = np.mean(reward_list)
        ave_reward_list.append(ave_reward)
        print('Episode {} Average Reward: {}'.format(i+1, ave_reward))
        reward_list = []






 #After creating the loop remember to CLOSE the environment of mountain car
env.close()


# --STEP 6--

# Plot Rewards

plt.plot(100*(np.arange(len(ave_reward_list)) + 1), ave_reward_list)
plt.xlabel('Episodes')
plt.ylabel('Average Reward')
plt.title('Average Reward vs Episodes')
plt.show()
