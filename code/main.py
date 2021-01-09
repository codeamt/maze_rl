from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from MazeEnv import MazeEnvironment
import argparse
import numpy as np

parser = argparse.ArgumentParser(
    description=""
)
parser.add_argument('epochs',
                    '--epochs',
                    type=int,
                    required=True,
                    default=500000,
                    help='Number of epochs for training the agent')

parser.add_argument('lr',
                    '--lr',
                    type=float,
                    required=True,
                    default=0.001,
                    help='Learning Rate for agent.')

parser.add_argument('world',
                    '--world',
                    type=list,
                    default=[[1, 0, 0, 2],
                             [0, 0, 0, 0],
                             [0, 3, 4, 3],
                             [0, 4, 0, 0]],
                    help='A list of 4 arrays, shape (1,4); Defaults to [[1,0,0,2], [[0,0,0,0]], [0,3,4,3], [ 0,4,0,0]]')

args = parser.parse_args()
epochs = args.epochs
lr = args.lr
world = args.world

maze_env = DummyVecEnv([lambda: MazeEnvironment(np.array(world))])
model = PPO2(MlpPolicy, maze_env, learning_rate=lr)
model.learn(epochs)

