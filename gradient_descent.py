from typing import Callable

def f(x: float) -> float:
    return (x - 3) ** 2

def g(x: float) -> float:
    return (x + 2) ** 2

def numeric_slope(fn: Callable[[float], float], x: float, h: float = 0.0001) -> float:
    return (fn(x + h) - fn(x)) / h

def gradient_descent(fn: Callable[[float], float], x: float, learning_rate: float = 0.1, steps: int = 20) -> float:
    for step in range(steps):
        slope = numeric_slope(fn, x)
        x = x - learning_rate * slope
        print(f"Step {step}: x = {x:.4f}, f(x) = {fn(x):.4f}")
    return x

print(numeric_slope(f, 0))
print(gradient_descent(f, 0))
print(gradient_descent(g, 1, 0.1, 40))