import numpy as np


class Quadratic(object):

    def predict(self, x):
        return np.square(x)
