import numpy as np
import matplotlib.pyplot as plt
import os

class Perceptron:
    """
    单层感知器神经网络模型
    支持多个输出神经元节点
    """
    def __init__(self):
        pass

    def train(self,
              X: np.ndarray,
              Y: np.ndarray,
              batch_size: int,
              u: float = 0.01,
              delta: float = 0.0001,
              epochs: int = 100,
              debug: bool = False):
        # 判断输入数据是否有效
        assert X.ndim == Y.ndim == 2 and X.shape[0] == Y.shape[0]
        
        # self.S：训练数据组数
        # self.M：输入信号数
        # self.N：输出神经元节点数
        self.S, self.M = X.shape
        _, self.N = Y.shape
        if debug:
            print(f"Data Size: {self.S}")
            print(f"M: {self.M}")
            print(f"N: {self.N}")

        # 参数和偏置矩阵随机化
        self.W = np.random.uniform(-1, 1, (self.M, self.N))
        self.B = np.random.uniform(-1, 1, (1, self.N))
        if debug:
            print(f"Initial W: \n{self.W}")
            print(f"Initial B: \n{self.B}")

        # 输入数据标准化
        # X_min = X.min(axis=1, keepdims=True)
        # X_max = X.max(axis=1, keepdims=True)
        # X = ((X - X_min) / (X_max - X_min)) * 2 - 1
        
        # 训练
        for epoch in range(epochs):
            loss = 0
            for batch_idx in range(int(self.S/batch_size)):
                X1: np.ndarray = X[(batch_idx * batch_size):((batch_idx + 1) * batch_size)]
                Y1: np.ndarray = Y[(batch_idx * batch_size):((batch_idx + 1) * batch_size)]
                T: np.ndarray = ((X1 @ self.W + self.B) > 0).astype(np.int32)
                E: np.ndarray = Y1 - T
                delta_W: np.ndarray = u * (X1.T @ E)
                delta_B: np.ndarray = u * np.sum(E, axis=0, keepdims=True)
                self.W += delta_W
                self.B += delta_B
                loss += np.linalg.norm(delta_W)
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss:.4f}')
            if loss < delta:
                break
        
    def __call__(self, X)->np.ndarray:
        return ((X @ self.W + self.B) > 0).astype(np.int32)

if __name__ == '__main__':
    print("Perceptron Test1:")
    X:np.ndarray = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    Y:np.ndarray = np.array([
        [0],
        [0],
        [0],
        [1]
    ])
    net = Perceptron()
    net.train(X, Y, 4)
    print(net(X))