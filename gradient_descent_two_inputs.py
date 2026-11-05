from typing import Callable

def subtract_vector(a: list[float], b: list[float]) -> list[float]:
    result: list[float] = []
    for i, j in zip(a, b):
        result.append(i - j)
    return result

def scale_vector(a: list[float], scalar: float) -> list[float]:
    result: list[float] = []
    for number in a:
        result.append(number * scalar)
    return result

def f(x: float, y: float) -> float:
    return (x - 1) ** 2 + (y - 2) ** 2

def numeric_gradient(fn: Callable[[float, float], float], x: float, y: float, h: float = 0.0001) -> list[float]:
    slope_x = (fn((x + h), y) - fn(x, y)) / h
    slope_y = (fn(x, (y + h)) - fn(x, y)) / h
    return [slope_x, slope_y]

def gradient_descent(fn: Callable[[float, float], float], point: list[float], learning_rate: float = 0.1, steps: int = 20) -> list[float]:
    for _ in range(steps):
        gradient = numeric_gradient(fn, point[0], point[1])
        point = subtract_vector(point, scale_vector(gradient, learning_rate))
    return point

if __name__ == "__main__":
    v1 = [2, 4]
    v2 = [3, 5]
    print(subtract_vector(v1, v2)) # [-1, -1]
    print([round(n, 4) for n in scale_vector(v2, 0.1)])   # [0.3, 0.5]
    result = gradient_descent(f, [5, 5], 0.1, 60)
    print(f"f: bottom at x = {result[0]:.4f} and y = {result[1]:.4f} (expected x = 1, y = 2), f(x, y) = {f(result[0], result[1]):.4f}")