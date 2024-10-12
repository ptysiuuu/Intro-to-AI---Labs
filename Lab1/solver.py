import torch


def solver(f, x0, step_size, epsilon):
    x0 = torch.tensor(x0, requires_grad=True)
    x = x0
    for n in range(10000):
        y = f(x)
        out = y.sum()
        out.backward()

        previous_x = x.clone()

        with torch.no_grad():
            x -= x.grad * step_size

        if (
            (torch.linalg.vector_norm(x.grad) <= epsilon)
            or
            (torch.linalg.vector_norm(x - previous_x) <= epsilon)
        ):
            return x

        x.grad.zero_()
    return None
