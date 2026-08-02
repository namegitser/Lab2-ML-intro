import numpy as np

def f(x):
    return (x - 3) ** 2

def grad_f(x):
    return 2 * (x - 3)

def gradient_descent_1d(start, lr, steps):
    history = [start]
    x = start
    for step in range(0, steps):
        x = x - lr * grad_f(x)
        history.append(x)
    return x, history
    

def f2(point):
    x, y = point
    return x**2 + 5*y**2

def grad_f2(point):
    x, y = point
    return np.array([2*x, 10*y])

def gradient_descent_2d(start, lr, steps):
    point = np.array(start, dtype=float)
    history = [point.copy()]

    for _ in range(steps):
        point = point - lr * grad_f2(point)
        history.append(point.copy())

    return point, np.array(history)


