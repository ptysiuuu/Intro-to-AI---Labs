import numpy as np
from itertools import product


class DataGenerator:
    def __init__(self):
        pass

    def get_train_data(self, domain, dim, fun, samples_density):
        domain_min = domain[0]
        domain_max = domain[1]
        x = np.linspace(domain_min, domain_max, samples_density)
        X = np.array(list(product(x, repeat=dim)))
        Y = fun(X)
        return X, Y

    def get_test_data(self, domain, dim, fun, samples_num):
        domain_min = domain[0]
        domain_max = domain[1]
        x = np.random.uniform(domain_min, domain_max, samples_num)
        X = np.array(list(product(x, repeat=dim)))
        Y = fun(X)
        return X, Y
