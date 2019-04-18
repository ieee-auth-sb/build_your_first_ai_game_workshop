# reinforcement-learning-game
This project contains a self trained ai game from open GymAI library using reinforecment learning that will be presented as a Machine Learning workshop for beginners at the Greek National ECE Student Conference (SFHMMY) of 2019.

## Installation
You need to install all dependencies first using
```bash
pip install -r requirements.txt
```
Install the GymAI games with,
```bash
pip install git+https://github.com/Kojoley/atari-py.git
```
## File list
* In `mountain_car.py` a Q-learning algorithm is used to train an agent to play and win the mountain car game, from gym, openAI library, as shown [here](https://gym.openai.com/envs/MountainCar-v0/).
 * In `space_invaders.py` the target is the same. An agent has to learn to win the Atari Space Invaders game. The game environment can be found [here](https://gym.openai.com/envs/SpaceInvaders-v0/). In this case, the environment is much more complex and a simple Q-learning algorithm can't solve quickly and easy enough. Deep learning is used in combination with a Q-table, written in Keras.
 * `check_env.py` is a simple script to check the existence and the real view of the space invaders game. It initializes the game and plays randomly forever, until the process is killed.
