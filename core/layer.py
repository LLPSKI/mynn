import numpy as np
from typing import Literal
from typing import Union
from typing import Tuple
from .utils import im2col
        

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
        

if __name__ == '__main__':
    pass
else:
    print("mynn.core.layer here!")