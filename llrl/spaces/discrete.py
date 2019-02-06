import numpy as np


class Discrete:
    def __init__(self, n):
        self.n = n

    def __repr__(self):
        return "Discrete(%d)" % self.n

    def __eq__(self, other):
        return self.n == other.n
        
    def sample(self):
        return np.random.randint(self.n)
        
    def shape(self):
        return (self.n,)

    def as_list(self):
        return list(range(self.n))
