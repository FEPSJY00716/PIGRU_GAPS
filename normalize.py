import numpy as np
import torch

class TrainData(object):
    def __init__(self):
        self.MaxValue = None
        self.MinValue = None
        self.range = None

    def _normal(self, data):
        normal_data = (data - self.MinValue) / self.MaxValue
        torch_data = normal_data
        return torch_data.clone().detach()

    def train_data(self, data):
        self.MinValue = data.min(axis=0).values
        self.MaxValue = data.max(axis=0).values
        self.range = self.MaxValue - self.MinValue
        return self._normal(data)

    def test_data(self, data):
        return self._normal(data)

class NoiseAdder:
    def __init__(self, noise_level):
        """
        初始化NoiseAdder类。
        :param noise_level: 噪声水平，例如0.3表示30%。
        """
        self.noise_level = noise_level

    def add_noise(self, data):
        """
        给数据添加噪声。
        :param data: 输入的PyTorch张量，尺寸为[672, 14]。
        :return: 添加了噪声的数据。
        """
        # 计算每个变量（列）的标准差
        stds = data.std(dim=0)

        # 计算噪声的标准差
        noise_stds = stds * self.noise_level

        # 为每个变量生成噪声
        noise = torch.randn_like(data) * noise_stds

        # 添加噪声到原始数据
        noisy_data = data + noise

        return noisy_data



class SlidingWindowDataGenerator:
    def __init__(self, window_size, step_size):
        self.window_size = window_size
        self.step_size = step_size

    def sliding_window(self, data):
        num_windows = (data.shape[0] - self.window_size) // self.step_size + 1
        windows = np.array(
            [data[i:i + self.window_size] for i in range(0, num_windows * self.step_size, self.step_size)])
        return windows

    def generate(self, data):
        sliding_windows = self.sliding_window(data)
        result = sliding_windows.reshape(-1, self.window_size, data.shape[1])
        return result


