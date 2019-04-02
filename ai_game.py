import retro, cv2, gym
import numpy as np

# Load game's environment
## DOWNLOAD MORE GAMES
def init():
    env = gym.make('SpaceInvaders-v0')
    # Check game's specs
    actions = env.action_space.n
    print(actions)
    print(env.observation_space.shape)
    # Q-Learning init
    Q = np.zeros([84, 84, actions])
    eta = .628
    gma = .9
    epis = 5000
    rev_list = [] # rewards per episode calculate
# Preprocess
# DOWNLOAD CV2
def preprocess(observation):
    observation = cv2.cvtColor(cv2.resize(observation, (84, 110)), cv2.COLOR_BGR2GRAY)
    observation = observation[26:110,:]
    ret, observation = cv2.threshold(observation,1,255,cv2.THRESH_BINARY)
    return np.reshape(observation,(84,84,1))

def init_network():
    pass

def train(epochs = 1000):
    init_network()
    for i in range(epochs):
        s = env.reset()
        # Reshape state
        s = preprocess(s)
        rAll = 0
        d = False
        j = 0
        while j < 99:
            env.render()
            j += 1
            a = np.argmax(Q[s,:] + np.random.randn(1,action_space)*(1./(i+1)))
            #Get new state & reward from environment
            s1,r,d,_ = env.step(a)
            s1 = preprocess(s1)
            #Update Q-Table with new knowledge
            Q[s,a] = Q[s,a] + eta*(r + gma*np.max(Q[s1,:]) - Q[s,a])
            rAll += r
            s = s1
            if d == True:
                break
        rev_list.append(rAll)
        env.render()
    print "Reward Sum on all episodes " + str(sum(rev_list)/epis)
    print "Final Values Q-Table"
    print Q


        # while True:
        #     # Keras Learning
        #
        #     # Next is BrainDQL case
        #     action = brain.getAction()
        #     actionmax = np.argmax(np.array(action))
        #
        #     nextObservation,reward, done, info = env.step(actionmax)
        #
        #     if done:
        #         nextObservation = env.reset()
        #     nextObservation = preprocess(nextObservation)
        #     brain.setPerception(nextObservation,action,reward,terminal)

    env.close()

if __name__ == '__main__':
    init()
    train()
