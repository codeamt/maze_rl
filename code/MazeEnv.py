from abc import ABC

import gym
from gym import spaces
import numpy as np


class MazeEnvironment(gym.Env, ABC):
    def __init__(self, world):
        """
        A Maze Environment for Reinforcement Learning.
        :param world (List[List[int]) - n*m grid maze.
        """
        self.action_space = spaces.Discrete(4)
        n = np.size(self.world_start, 0)
        m = np.size(self.world_start, 1)
        self.observation_space = spaces.Box(low=0,
                                            high=4,
                                            shape=(n + 1, m),
                                            dtype=np.int16)
        self.reward_range = (-200, 200)
        self.current_episode = 0
        self.success_episode = []
        self.world_start = np.array(world)
        self.world = None
        self.current_player = None
        self.state = None
        self.current_step = 0
        self.max_step = 30
        self.explore_incentive = None
        self.bonus_reward = 0

    def reset(self):
        """
        Env Reset Method.
        :return:
        """
        self.current_player = 1
        self.state = 'P'
        self.current_step = 0
        self.max_step = 30
        self.world = np.copy(self.world_start)
        self.explore_incentive = np.ones(
            shape=(np.size(self.world, 0),
                   np.size(self.world, 1)))
        self.bonus_reward = 0
        return self._next_observation()

    def step(self, action):
        """
        Env Step Method.
        :param action:
        :return (tuple) -> (obs, reward, done, state)
        """
        self._take_action(action)
        self.current_step += 1
        reward = None
        done = False
        print(self.world)
        if self.state == "W":
            print(f"Player {self.current_player} won.")
            reward = 100 * (1 + 1 / self.current_step)
            done = True
        elif self.state == "L":
            print(f"Player {self.current_player} lost.")
            reward = -200
            done = True
        elif self.state == "P":
            reward = -2

        if self.current_step >= self.max_step:
            print(f'New episode number {self.current_episode + 1}')
            done = True

        if self.current_player == 1:
            self.current_player = 2
        else:
            self.current_player = 1

        reward += self.bonus_reward
        self.bonus_reward = 0

        if done:
            self.render_episode(self.state)
            self.current_episode += 1

        obs = self._next_observation()
        return obs, reward, done, {'state': self.state}

    def render_episode(self, win_or_lose):
        """
        Env Render Method.
        :param win_or_lose (bool)
        """
        self.success_episode.append(
            'Success' if win_or_lose == 'W' else 'Failure'
        )
        file = open("render/render.txt", 'a')
        file.write('--------------------------\n')
        file.write(f"Episode number {self.current_episode}\n")
        file.write(f"{self.success_episode[-1]} in {self.current_step} steps.\n")
        file.close()

    def _exploration_prize(self, next_pos):
        """
        Incentive mechanism for exploration.
        :param next_pos (int):

        """
        if self.explore_incentive[next_pos] == 1:
            self.explore_incentive[next_pos] = 0
            self.bonus_reward += 1

    def _next_observation(self):
        """
        Helper Method for Getting the Next Observation.
        :return: obs (np.array) -> next player and self.world, flattened.
        """
        obs = self.world
        data_to_add = [0] * np.size(self.world, 1)
        data_to_add[0] = self.current_player
        obs = np.append(obs, [data_to_add], axis=0)
        return obs

    def _teleport(self):
        """
        Teleport helper method.
        """
        other_player = 2 if self.current_player == 1 else 1
        other_player_pos = np.where(self.world == other_player)
        other_next_pos = (other_player_pos[0] + 3, other_player_pos[1])
        if other_next_pos[0] < np.size(self.world, 0):
            self.world[other_next_pos] = other_player
            self.world[other_next_pos] = 0

    def _take_action(self, action):
        """
        Helper Method for Determining the Next Action
        :param action (int).
        """
        current_pos = np.where(self.world == self.current_player)

        if action == 0:
            next_pos = (current_pos[0] - 1, current_pos[1])
            self.world[next_pos] = self.current_player
            self.world[current_pos] = 0

            if next_pos[0] >= 0 and int(self.world[next_pos]) == 0:
                self._exploration_prize(next_pos)

            elif next_pos[0] >= 0 and int(self.world[next_pos]) in (1, 2):
                pass

            elif next_pos[0] >= 0 and (int(self.world[next_pos]) == 3):
                self.state = 'L'
                self._exploration_prize(next_pos)

            elif next_pos[0] >= 0 and (int(self.world[next_pos]) == 4):
                self._teleport()
                self.state = 'P'
                self._exploration_prize(next_pos)

            elif next_pos[0] >= 0 and (int(self.world[next_pos]) == 5):
                self.state = 'W'
                self._exploration_prize(next_pos)

        elif action == 1:
            next_pos = (current_pos[0], current_pos[1] + 1)
            limit = np.size(self.world, 1)
            self.world[next_pos] = self.current_player
            self.world[current_pos] = 0

            if next_pos[1] < limit and int(self.world[next_pos]) == 0:
                self._exploration_prize(next_pos)

            elif next_pos[1] < limit and int(self.world[next_pos]) in (1, 2):
                pass

            elif next_pos[1] < limit and (int(self.world[next_pos]) == 3):
                self.state = "L"
                self._exploration_prize(next_pos)

            elif next_pos[1] < limit and (int(self.world[next_pos]) == 4):
                self._teleport()
                self.state = 'P'
                self._exploration_prize(next_pos)

            elif next_pos[1] < limit and (int(self.world[next_pos]) == 5):
                self.state = 'W'
                self._exploration_prize(next_pos)

        elif action == 2:
            next_pos = (current_pos[0] + 1, current_pos[1])
            limit = np.size(self.world, 0)
            self.world[next_pos] = self.current_player
            self.world[current_pos] = 0

            if next_pos[0] < limit and int(self.world[next_pos]) == 0:
                self._exploration_prize(next_pos)

            elif next_pos[0] < limit and int(self.world[next_pos]) in (1, 2):
                pass

            elif next_pos[0] < limit and (int(self.world[next_pos]) == 3):
                self.state = "L"
                self._exploration_prize(next_pos)

            elif next_pos[0] < limit and (int(self.world[next_pos]) == 4):
                self._teleport()
                self.state = 'P'
                self._exploration_prize(next_pos)

            elif next_pos[0] < limit and (int(self.world[next_pos]) == 5):
                self.state = 'W'
                self._exploration_prize(next_pos)

        elif action == 3:
            next_pos = (current_pos[0], current_pos[1] - 1)
            self.world[next_pos] = self.current_player
            self.world[current_pos] = 0

            if next_pos[1] >= 0 and int(self.world[next_pos]) == 0:
                self._exploration_prize(next_pos)

            elif next_pos[1] >= 0 and int(self.world[next_pos]) in (1, 2):
                pass

            elif next_pos[1] >= 0 and (int(self.world[next_pos]) == 3):
                self.state = "L"
                self._exploration_prize(next_pos)

            elif next_pos[1] >= 0 and (int(self.world[next_pos]) == 4):
                self._teleport()
                self.state = 'P'
                self._exploration_prize(next_pos)

            elif next_pos[1] >= 0 and (int(self.world[next_pos]) == 5):
                self.state = 'W'
                self._exploration_prize(next_pos)
