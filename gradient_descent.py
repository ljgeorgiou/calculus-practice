from typing import Callable

def f(x: float) -> float:
    return (x - 3) ** 2

def g(x: float) -> float:
    return (x + 2) ** 2

def k(x: float) -> float:
    return (x - 5) ** 2

def numeric_slope(fn: Callable[[float], float], x: float, h: float = 0.0001) -> float:
    return (fn(x + h) - fn(x)) / h

def gradient_descent(fn: Callable[[float], float], x: float, learning_rate: float = 0.1, steps: int = 20) -> float:
    for _ in range(steps):
        slope = numeric_slope(fn, x)
        x = x - learning_rate * slope
    return x

if __name__ == "__main__":
    print(f"slope of f at 0 = {numeric_slope(f, 0):.4f} (expected about -6)")
    result = gradient_descent(f, 0)
    print(f"f: bottom found at x = {result:.4f} (expected 3), f(x) = {f(result):.4f}")
    result = gradient_descent(g, 1, 0.1, 40)
    print(f"g: bottom found at x = {result:.4f} (expected -2), g(x) = {g(result):.4f}")
    result = gradient_descent(k, 3)
    print(f"k: bottom found at x = {result:.4f} (expected 5), k(x) = {k(result):.4f}")