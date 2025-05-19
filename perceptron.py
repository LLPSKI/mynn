import numpy as np
import matplotlib.pyplot as plt
import os

class Perceptron:
    def __init__(self):
        pass

    def train(self,
              X: np.ndarray,
              Y: np.ndarray,
              u: float = 0.01,
              delta: float = 0.01,
              epochs: int = 50,
              debug: bool = False,
              debug_dir: str = None):
        # 判断输入数据是否有效
        assert X.ndim == Y.ndim == 2 and X.shape[1] == Y.shape[1]
        self.M, self.P = X.shape
        self.N, _ = Y.shape

        # 支持2维平面（两个参数）中一条直线（一个输出神经元）
        if self.M != 2 or self.N != 1:
            debug = False

        # 参数和偏置矩阵随机化
        self.W = np.random.uniform(-1, 1, (self.N, self.M))
        self.B = np.random.uniform(-1, 1, (self.N, 1))

        # 输入数据标准化
        X_min = X.min(axis=1, keepdims=True)
        X_max = X.max(axis=1, keepdims=True)
        X = ((X - X_min) / (X_max - X_min)) * 2 - 1

        # 作图
        if debug:
            fig, axes = plt.subplots(4, 1, figsize=(8, 16))
            x1 = X[0, :]
            x2 = X[1, :]
            x: np.ndarray = np.linspace(-1.5, 1.5, 1000)
            y = Y[0, :]
            mask = (y == 1)
            for ax in axes:
                # 样本点标记，为1的'*'，为0的'o'
                ax.plot(x1[mask], x2[mask], '*')
                ax.plot(x1[~mask], x2[~mask], 'o')
                ax.set_xlabel('x1')
                ax.set_ylabel('x2')
                ax.set_xlim(-1.5, 1.5)
                ax.set_ylim(-1.5, 1.5)
                ax.grid(True)
            tp = 0 # 第tp张图
            sub = 0 # 第sub号子图
        
        # 训练
        for epoch in range(epochs):
            T: np.ndarray = ((self.W @ X + self.B) > 0).astype(np.int32)
            E: np.ndarray = Y - T
            delta_W: np.ndarray = u * (E @ X.T)
            delta_B: np.ndarray = u * np.sum(E, axis=1, keepdims=True)
            self.W += delta_W
            self.B += delta_B
            delta_n = np.linalg.norm(delta_W)

            # 作图
            if debug:
                w1 = self.W[0][0]
                w2 = self.W[0][1]
                b = self.B[0][0]
                y = [-(w1 / w2) * i - (b / w2) for i in x]
                axes[sub].plot(x, y, '-k', label=f"delta={delta_n}")
                axes[sub].set_title(f'epoch={epoch}')
                axes[sub].legend()
                if sub == 3:
                    # 一张图绘制完了
                    plt.tight_layout()
                    fig.savefig(f'{debug_dir}\\{tp}.png', dpi=400)
                    sub = 0
                    tp += 1
                    for ax in axes:
                        ax.lines[2].remove()
                        ax.legend_.remove()
                        ax.set_title('')
                else:
                    sub += 1
            if delta_n < delta:
                break
        
        # 作图
        if debug and sub > 0:
            plt.tight_layout()
            fig.savefig(f'{debug_dir}\\{tp}.png', dpi=400)
        
    def test(self,
             X: np.ndarray) -> np.ndarray:
        return (np.dot(self.W, X) > 0).astype(np.int32)

    def predict(self,
                X: np.ndarray) -> np.ndarray:
        return ((self.W @ X + self.B) > 0).astype(np.int32)
        
if __name__ == '__main__':
    # 测试
    script_dir = os.path.dirname(__file__)
    debug_dir = f'{script_dir}\\debug'
    for file in os.listdir(debug_dir):
        os.remove(os.path.join(debug_dir, file))
    
    # 正样本
    x1 = np.random.uniform(-10, 11, (100, 1))
    x2 = 3 * x1 + (100000)
    X = np.hstack((x1, x2))
    Y = np.random.randint(1, 2, (100, 1))

    # 负样本
    x1 = np.random.uniform(-10, 11, (100, 1))
    x2 = 3 * x1 + (100001)
    X = np.vstack((X, np.hstack((x1, x2))))
    Y = np.vstack((Y, np.random.randint(0, 1, (100, 1))))
    
    net = Perceptron()
    net.train(X.T, Y.T, debug=True, debug_dir=f'{script_dir}\\debug')