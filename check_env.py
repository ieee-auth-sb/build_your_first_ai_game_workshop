import retro

def main():
    env = retro.make(game="Airstriker-Genesis")
    observation = env.reset()
    while True:
        observation, reward, done, info = env.step(env.action_space.sample())
        env.render()
        if done:
            observation = env.reset()
    env.close()

if __name__ == '__main__':
    main()
