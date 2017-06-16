import numpy as np
from scipy.signal import convolve2d
import matplotlib.animation as animation


def update_board(X):
    # Check out the details at: https://jakevdp.github.io/blog/2013/08/07/conways-game-of-life/
    # Compute number of neighbours,
    N = convolve2d(X, np.ones((3, 3)), mode='same', boundary='wrap') - X

    # Apply rules of the game
    X = (N == 3) | (X & (N == 2))
    return X

