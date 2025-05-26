import numpy as np

def cross_entropy_loss(X:np.ndarray,
                       Y:np.ndarray) -> np.ndarray:
    Y_mask = (Y == 1)
    return -np.log(X[Y_mask])