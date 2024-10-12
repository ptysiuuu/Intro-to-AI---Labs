import torch
from solver import solver


def f(x):
    return torch.sin(x)  # example function for debugging


def main():
    x0 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    result = solver(f, x0, 1, 1e-10)
    print(result)


if __name__ == "__main__":
    main()
