from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from MazeEnv import MazeEnvironment
import argparse
import numpy as np
from typing import List
import time

parser = argparse.ArgumentParser(
    description=""
)
parser.add_argument('epochs',
                    '--epochs',
                    type=int,
                    required=True,
                    default=1000000,
                    help='Number of epochs for training the agent. Default: 1000000.')

parser.add_argument('lr',
                    '--lr',
                    type=float,
                    required=True,
                    default=0.001,
                    help='Learning Rate for agent. Default: 0.001')

parser.add_argument('gamma',
                    '--gamma',
                    type=float,
                    required=True,
                    default=0.000001,
                    help='Gamma (Discount Factor). Default: 0.000001.')

parser.add_argument('lam',
                    '--lam',
                    type=float,
                    required=True,
                    default=0,
                    help='Lambda (aka GAE Parameter). Default: 0.')

parser.add_argument('world',
                    '--world',
                    type=List[List[int]],
                    default=[[1, 0, 2, 0, 0],
                             [0, 0, 0, 0, 0],
                             [4, 0, 0, 0, 0],
                             [0, 0, 3, 3, 0],
                             [0, 0, 3, 5, 0]],
                    help='A (2D) array, shape (n,m); Defaults: [[1, 0, 2, 0, 0], [0, 0, 0, 0, 0], [4, 0, 0, 0, 0], '
                         '[0, 0, 3, 3, 0], [0, 0, 3, 5, 0]]')

parser.add_argument('inference',
                    '--inference',
                    type=bool,
                    required=True,
                    default=True,
                    help='Whether or not to test the agent after training. Default: True.')

args = parser.parse_args()
epochs = args.epochs
lr = args.lr
gamma = args.gamma
lam = args.lam
world = args.world
inference = args.inference

maze_env = DummyVecEnv([lambda: MazeEnvironment(np.array(world))])
model = PPO2(MlpPolicy,
             maze_env,
             learning_rate=lr,
             gamma=gamma,
             lam=lam)
print(f"Training agents for {epochs//30} episodes.")
start = time.time()
model.learn(epochs)
end = time.time()

if inference is True:
    print(f"Training complete. Duration: {end-start} seconds.")
    print(f"Testing results. Getting Number of Wins for 300 steps...")
    result_test = []
    obs = maze_env.reset()
    for i in range(300):
        action, _states = model.predict(obs)
        obs, reward, done, info = maze_env.step(action)
        if done:
            result_test.append(info[0]['state'])
    score = result_test.count('W')/len(result_test)
    print(f"Inference test complete. Performance score: {score * 100} %")

