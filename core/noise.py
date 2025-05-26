import numpy as np

def add_salt_pepper_noise(data:np.ndarray, salt_prob=0.05, pepper_prob=0.05):
    """
    添加椒盐噪声（向量化实现）
    参数:
        data: 输入数据
        salt_prob: 盐噪声概率，置1
        pepper_prob: 椒噪声概率，置0
    返回:
        加噪后的数据
    """
    noisy_data = data.copy()
    shape = data.shape
    # 生成随机掩码
    mask = np.random.choice([0, 1, 2], size=shape, p=[pepper_prob, salt_prob, 1 - salt_prob - pepper_prob])
    # 应用噪声
    noisy_data[mask == 0] = 0  # 椒噪声
    noisy_data[mask == 1] = 1  # 盐噪声
    return noisy_data