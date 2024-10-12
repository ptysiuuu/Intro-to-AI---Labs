import torch
from solver import solver


def f(x):
    return torch.sin(x)  # example function for debugging


def main():
    x0 = torch.rand(10, 1, requires_grad=True)
    result, n = solver(f, x0, 1, 0.01)
    print(result, n)


if __name__ == "__main__":
    main()
