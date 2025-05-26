import numpy as np
from typing import Literal
from typing import Union
from typing import Tuple
from .utils import im2col

class Linear:
    """
    全连接层实现
    """
    def __init__(self,
                 m:int,
                 n:int):
        self.M, self.N = m, n
        self.W = np.random.uniform(-1, 1, (self.N, self.M))
        self.B = np.random.uniform(-1, 1, (self.N, 1))
    
    def forward(self,
                X:np.ndarray) -> np.ndarray:
        assert X.shape[0] == self.M
        self.X = X
        return (self.W @ X) + self.B
    
    def backward(self,
                 dout:np.ndarray) -> np.ndarray:
        assert dout.shape[0] == self.N
        self.dW = dout @ self.X.T
        self.dB = np.sum(dout, axis=1)
        dX = self.W.T @ dout

        return dX
    
class ReLU:
    """
    ReLU层实现
    """
    def __init__(self):
        pass

    def forward(self,
                X:np.ndarray) -> np.ndarray:
        self.X = X
        return np.maximum(0, X)
    
    def backward(self,
                 dout:np.ndarray) -> np.ndarray:
        return (dout > 0).astype(np.float64)

class Step:
    """
    阶跃函数层/感知器输出层实现
    """
    def __init__(self):
        pass
    
    def forward(self,
                X:np.ndarray) -> np.ndarray:
        return (X > 0).astype(np.int32)
    
    def backward(self,
                 dout:np.ndarray) -> np.ndarray:
        return dout
    
class Softmax:
    """
    Softmax层实现
    """
    def __init__(self):
        pass

    def forward(self,
                X:np.ndarray) -> np.ndarray:
        exp_X = np.exp(X)
        self.Y = exp_X / np.sum(exp_X, axis=0)
        return self.Y
    
    def backward(self,
                 dout:np.ndarray) -> np.ndarray:
        return self.Y - dout

class Convolution:
    def __init__(self, 
                 W: np.ndarray,
                 b: np.ndarray,
                 stride: int = 1,
                 pad_mode: Union[Tuple[int, int], 
                           Literal['valid', 'auto']] 
                           = 'auto'):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad_mode = pad_mode
    
    def forward(self, 
                X: np.ndarray) -> np.ndarray: 
        FN, C, filter_h, filter_w = self.W.shape
        N, C, H, W = X.shape
        col = im2col(X, filter_h, filter_w, self.stride, self.pad_mode)
        col_W = self.W.reshape(FN, -1).transpose(1, 0)
        out: np.ndarray = np.dot(col, col_W) + self.b
        return out.reshape(N, H, W, -1).transpose(0, 3, 1, 2)