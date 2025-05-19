import numpy as np
from typing import Literal
from typing import Union
from typing import Tuple

def im2col(input_data: np.ndarray,
           filter_h: int,
           filter_w: int,
           stride: int = 1,
           pad_mode: Union[Tuple[int, int], 
                           Literal['valid', 'auto']] 
                           = 'valid') -> np.ndarray:
    """
    将4D输入数据展开为2D矩阵以便卷积运算
    
    参数:
        input_data : 形状为 (N, C, H, W) 的4D数组
        filter_h : 滤波器高度
        filter_w : 滤波器宽度
        stride : 卷积步长
        pad : 填充模式，支持以下选项：
            - 'valid'（默认）：不填充，输出尺寸减小
            - 'auto'：自动填充使输出尺寸与输入相同
            - Tuple：二维元组 (pad_h, pad_w)
        
    返回:
        形状为 (N*out_h*out_w, C*filter_h*filter_w) 的2D数组
    """
    N, C, H, W = input_data.shape

    # 计算填充量
    if pad_mode == 'valid':
        pad_h = pad_w = 0
    elif pad_mode == 'auto':
        pad_h = ((H - 1) * stride - H + filter_h) // 2
        pad_w = ((W - 1) * stride - W + filter_w) // 2
        print(pad_h, pad_w)
    else:
        pad_h = pad_mode[0]
        pad_w = pad_mode[1]
    
    out_h = (H + 2 * pad_h - filter_h) // stride + 1
    out_w = (W + 2 * pad_w - filter_w) // stride + 1

    # 填充输入数据
    img = np.pad(input_data, [(0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)])
    
    # 创建输出矩阵
    col = np.zeros((N * out_h * out_w, C * filter_h * filter_w))
    for n in range(N):
        for w in range(out_w):
            for h in range(out_h):
                patch = img[n, 
                            :, 
                            h * stride : h * stride + filter_h,
                            w * stride : w * stride + filter_w]
                col[n * out_h * out_w + w * out_w + h] = patch.reshape(-1)
    
    return col

if __name__ == '__main__':
    pass
else:
    print("mynn.core.utils here!")