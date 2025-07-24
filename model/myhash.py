import numpy as np
import torch.distributions


class MyHash(object):
    def __init__(self, num_embedding: int):
        self.hash = [0] * num_embedding

    def count(self, states: np.ndarray):
        counts = []
        states = states.reshape(states.shape[0])
        for state in states:
            key = int(state)
            self.hash[key] += 1
            counts.append(self.hash[key])

        return np.array(counts)


class DreamerHash:
    def __init__(self, n_categorical: int, n_classes: int):
        self.n_categorical = n_categorical
        self.n_classes = n_classes

        self.hash = [0] * (n_classes ** n_categorical)
        self.A = np.array([n_classes ** i for i in range(n_categorical)])

        self.teacher_baseline_hash = None

    def get_index(self, states: np.ndarray, _type: str):
        counts = []
        nums = []
        batch_size, episode_length, n_cate, n_classes = states.shape
        states_index = np.argmax(states, axis=3)
        states_index = states_index.reshape((states_index.shape[0] * states_index.shape[1], -1))
        for state_index in states_index:
            num = (self.A * state_index).sum()
            nums.append(num)
            if self.hash[num] > 0:
                counts.append(self.hash[num])
            else:
                counts.append(1)

        counts = np.array(counts)

        if _type == "c*c*0.005":
            normed_counts = np.power(counts, 2) * 0.005

            factor = 1 / normed_counts
            limited_factor = np.where(factor > 1, 1, factor)

        elif _type == "cut_off_at_1000":
            factor = 1 / counts
            limited_factor = np.where(factor < 0.001, 0, np.ones_like(factor))

        elif _type == "cut_off_at_100":
            factor = 1 / counts
            limited_factor = np.where(factor < 0.01, 0, np.ones_like(factor))

        elif _type == "cut_off_at_10000":
            factor = 1 / counts
            limited_factor = np.where(factor < 0.0001, 0, np.ones_like(factor))

        elif _type == "sqrt":
            factor = 1 / counts
            limited_factor = np.sqrt(factor)
        elif _type == "sqrt*10":
            factor = 1 / counts
            limited_factor = np.sqrt(factor * 10)
        elif _type == "1/n":
            limited_factor = 1 / counts
        elif _type == "1/n*10":
            limited_factor = 1 / counts * 10
        elif _type == "1/n*100":
            limited_factor = 1 / counts * 100
        elif _type == "1/n*1000":
            limited_factor = 1 / counts * 1000
        else:
            raise RuntimeError

        # normed_factor = np.pow(counts, 2)
        limited_factor = np.where(limited_factor > 1, 1, limited_factor)

        m = torch.distributions.Bernoulli(torch.tensor(limited_factor))
        mask = m.sample()

        for result, num in zip(mask, nums):
            if result == 1:
                self.hash[num] += 1

        return (np.array(mask).reshape((batch_size, episode_length)),
                np.array(counts).reshape((batch_size, episode_length)))

    def get_index_with_teacher_baseline(self, states: np.ndarray):
        if self.teacher_baseline_hash is None:
            raise RuntimeError

        results = []
        nums = []
        batch_size, episode_length, n_cate, n_classes = states.shape
        states_index = np.argmax(states, axis=3)
        states_index = states_index.reshape((states_index.shape[0] * states_index.shape[1], -1))
        for state_index in states_index:
            num = (self.A * state_index).sum()
            nums.append(num)

            count = self.hash[num]
            results.append(1.0 if count <= self.teacher_baseline_hash[num] else 0.0)

        for result, num in zip(results, nums):
            if result == 1:
                self.hash[num] += 1

        return np.array(results).reshape((batch_size, episode_length))

    def count(self, states: np.ndarray):
        counts = []
        batch_size, episode_length, n_cate, n_classes = states.shape
        states_index = np.argmax(states, axis=3)
        states_index = states_index.reshape((states_index.shape[0] * states_index.shape[1], -1))
        for state_index in states_index:
            num = (self.A * state_index).sum()
            self.hash[num] += 1
            counts.append(self.hash[num])

        return np.array(counts).reshape((batch_size, episode_length))

    def clear(self):
        self.hash = [0] * (self.n_classes ** self.n_categorical)

    def to_npy(self, file_name: str) -> None:
        np.array(self.hash).tofile(file_name)
