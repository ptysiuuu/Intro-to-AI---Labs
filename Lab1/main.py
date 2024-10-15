from solver import solver
from cec2017.functions import f3, f12
import autograd.numpy as np
from matplotlib import pyplot as plt


def quadriatic(x):
    return np.sum(x ** 2)


def main():

    x0 = np.random.uniform(-100, 100, 10)
    result, iterations, values = solver(f3, x0, 0.000001, 1e-4, 1000, 100)
    print(result)
    print(quadriatic(result))
    plt.plot(range(iterations), values)
    plt.show()
    print(values)


if __name__ == "__main__":
    main()
