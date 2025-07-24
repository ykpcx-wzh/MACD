import random
from typing import Tuple

import numpy as np
from enum import IntEnum


class BlockType(IntEnum):
    OutSide = 0
    Empty = 1
    Agent = 2
    Goal = 3


class Move(IntEnum):
    Up = 0
    Down = 1
    Left = 2
    Right = 3
    Stay = 4


class Agent:
    def __init__(self, x: int, y: int, _id: int):
        self.x = x
        self.y = y
        self._id = _id
        self.arrived: bool = False


class FindOneGoal:
    def __init__(
        self,
        map_size: int,
        agent_num: int,
        seed: int = 999,
        sight: int = 2,
        episode_limit: int = 200,
    ):
        super(FindOneGoal, self).__init__()
        self.map_size = map_size
        self.agent_num = agent_num

        self._seed = seed
        self._sight = sight
        self._map = None
        self._vision_map = None

        self.n_agents = agent_num
        self.n_actions = len(Move)
        self.episode_limit = episode_limit

        self.agents: dict[int, Agent] = {}
        self.goal_pos = [0, 0]
        self.pos_length = len(bin(self.map_size)) - 2

        self._episode_steps = 0
        self.terminated = False
        self.truncated = False
        self.win_counted = False

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        self._map = (
            np.zeros(shape=(self.map_size, self.map_size), dtype=np.int8)
            + BlockType.Empty.value
        )
        self.agents.clear()
        self._episode_steps = 0
        self.terminated = False
        self.truncated = False
        self.win_counted = False

        for i in range(self.agent_num):
            pos_x, pos_y = random.randint(0, self.map_size - 1), random.randint(
                0, self.map_size - 1
            )
            self.agents[i] = Agent(pos_x, pos_y, i)
            self._map[pos_x, pos_y] = BlockType.Agent.value

        pos_x, pos_y = random.randint(0, self.map_size - 1), random.randint(
            0, self.map_size - 1
        )
        self._map[pos_x, pos_y] = BlockType.Goal.value
        self.goal_pos = [pos_x, pos_y]

        self._vision_map = np.pad(
            self._map, self._sight, "constant", constant_values=BlockType.OutSide.value
        )

        return self.get_state(), self.get_obs()

    def _agent_take_action(self, agent: Agent, action: int) -> None:

        if agent.arrived:
            return

        if action == Move.Stay.value:
            pass
        elif action == Move.Up.value:
            if agent.y > 0:
                self._map[agent.x, agent.y] = BlockType.Empty.value
                agent.y -= 1
                self._map[agent.x, agent.y] = BlockType.Agent.value
        elif action == Move.Down.value:
            if agent.y < self.map_size - 1:
                self._map[agent.x, agent.y] = BlockType.Empty.value
                agent.y += 1
                self._map[agent.x, agent.y] = BlockType.Agent.value
        elif action == Move.Left.value:
            if agent.x > 0:
                self._map[agent.x, agent.y] = BlockType.Empty.value
                agent.x -= 1
                self._map[agent.x, agent.y] = BlockType.Agent.value
        elif action == Move.Right.value:
            if agent.x < self.map_size - 1:
                self._map[agent.x, agent.y] = BlockType.Empty.value
                agent.x += 1
                self._map[agent.x, agent.y] = BlockType.Agent.value
        else:
            raise ValueError

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float, bool, bool]:
        if self.terminated or self.truncated:
            raise RuntimeError("游戏已结束！！！！")

        for action, agent in zip(actions, self.agents.values()):
            self._agent_take_action(agent, action)

        reward = self.get_reward()

        self._episode_steps += 1

        if self._episode_steps >= self.episode_limit:
            self.truncated = True

        # if self.win_counted:
        #     self.terminated = True

        return self.get_state(), self.get_obs(), reward, self.terminated, self.truncated

    def get_reward(self) -> float:
        reward = 0
        win: bool = True
        for _, agent in self.agents.items():
            if abs(agent.x - self.goal_pos[0]) + abs(agent.y - self.goal_pos[1]) <= 1:
                if not agent.arrived:
                    reward += 1
                    agent.arrived = True
            else:
                reward -= 0.05
                win = False

        self.win_counted = win
        # if win:
        #     reward += 100

        return reward

    def get_obs_agent(self, agent_id: int) -> np.ndarray:
        agent = self.agents[agent_id]
        slice_x = slice(agent.x, agent.x + self._sight * 2 + 1)
        slice_y = slice(agent.y, agent.y + self._sight * 2 + 1)
        _agent_sight_obs = self._vision_map[slice_x, slice_y]
        _agent_pos = [agent.x, agent.y]
        _agent_obs = np.append(_agent_sight_obs, _agent_pos)

        return _agent_obs

    def get_obs(self) -> np.ndarray:
        self._map[self.goal_pos[0], self.goal_pos[1]] = BlockType.Goal.value
        all_obs = []
        for i in range(self.agent_num):
            agent_obs = self.get_obs_agent(i)
            all_obs.append(agent_obs)

        return np.array(all_obs)

    def seed(self) -> int:
        return self._seed

    def get_state(self) -> np.ndarray:
        return self._map.flatten()

    def get_state_size(self) -> int:
        return self.map_size * self.map_size

    def get_obs_size(self) -> int:
        return (self._sight * 2 + 1) ** 2 + 2

    def get_available_actions(self) -> np.ndarray:
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)

        return np.array(avail_actions)

    def get_avail_agent_actions(self, agent_id: int) -> np.ndarray:
        avail_actions = [0] * self.n_actions
        agent = self.agents[agent_id]
        for move in [Move.Up, Move.Down, Move.Left, Move.Right, Move.Stay]:
            if self._can_move(agent, move):
                avail_actions[move.value] = 1

        return np.array(avail_actions)

    def get_total_actions(self) -> int:
        return self.n_actions

    def _can_move(self, agent: Agent, action: Move) -> bool:
        if action == Move.Stay:
            res = True
        elif action == Move.Up:
            if agent.y > 0:
                res = True
            else:
                res = False
        elif action == Move.Down:
            if agent.y < self.map_size - 1:
                res = True
            else:
                res = False
        elif action == Move.Left:
            if agent.x > 0:
                res = True
            else:
                res = False
        elif action == Move.Right:
            if agent.x < self.map_size - 1:
                res = True
            else:
                res = False
        else:
            raise ValueError

        return res

    @property
    def obs_shape(self) -> int:
        return self.get_obs_size()

    @property
    def state_shape(self) -> int:
        return self.get_state_size()

    def get_sight(self):
        return self._sight


if __name__ == "__main__":
    env = FindOneGoal(map_size=10, agent_num=2)
    state, obs = env.reset()
    print(env._map)
    print(np.array(obs[0][0:25]).reshape((5, 5)))
    print(np.array(obs[1][0:25]).reshape((5, 5)))

    acts = env.get_available_actions()
    print(acts)
