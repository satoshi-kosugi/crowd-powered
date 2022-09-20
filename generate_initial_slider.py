import numpy as np
from typing import Tuple

class InitialSliderGenerator:

    def __init__(self, num_dims):
        self.end_0 = np.random.uniform(low=0.0, high=1.0, size=(num_dims, ))
        boundary = np.sign(0.5 - self.end_0) * 0.5 + 0.5
        ratio = np.abs((boundary - 0.5) / (0.5 - self.end_0 + 0.000001)).min()
        self.end_1 = (0.5 - self.end_0) * ratio +  0.5

    def generate_initial_slider(self, num_dims: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.end_0, self.end_1
