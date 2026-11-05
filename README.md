# calculus-practice

Gradient descent built from scratch in plain Python, with no NumPy. Gradient descent is the method almost every AI model uses to learn, so I wanted to build it myself and understand how it really works.

## What's in here

```
gradient_descent.py               gradient descent with one input, tested on three different functions
gradient_descent_two_inputs.py    gradient descent with two inputs, using vectors
```

## What is a derivative?

The derivative tells you the gradient (the slope) of a curve at any point, so how steep it is there. For example:

- A gradient of 2 means for every 1 step to the right, we go up by 2.
- A gradient of 0.5 means for every 1 step to the right, we go up by 0.5.
- A gradient of -3 means for every 1 step to the right, we go down by 3.

So the sign tells us the direction (positive means going up, negative means going down), and the size of the number tells us how steep it is.

The derivative is a formula, and you plug in a point to get the gradient at that point. If we have the function `f(x) = (x - 3)²`, the derivative is `f'(x) = 2x - 6`. Now watch what the derivative says at different spots:

- At `x = 0`: `f'(0) = 2(0) - 6 = -6`. The gradient is negative, so the curve is going down here, and it's steep: for every 1 step to the right, we go down by 6.
- At `x = 5`: `f'(5) = 2(5) - 6 = 4`. The gradient is positive, so the curve is going up here: for every 1 step to the right, we go up by 4.
- At `x = 3`: `f'(3) = 2(3) - 6 = 0`. The gradient is 0, so the curve is flat here. This is the bottom of the curve.

## What is the chain rule for?

The chain rule is for finding the derivative of a function that's inside another function. For example, in `(2x + 1)²`, the inner function is `2x + 1`, and the outer function is "square it". The rule is:

```
derivative = (derivative of the outer, with the inner still inside) times (derivative of the inner)
```

**Example 1:** `f(x) = (2x + 1)²`

- The derivative of the outer, with the inner still inside: "square something" becomes `2 × (something)`, so we get `2(2x + 1)`.
- The derivative of the inner, `2x + 1`, is `2`.

Therefore:

```
f'(x) = 2(2x + 1) × 2
f'(x) = 4(2x + 1)
f'(x) = 8x + 4
```

**Example 2:** `f(x) = (x - 3)²`

- The derivative of the outer, with the inner still inside: `2(x - 3)`.
- The derivative of the inner, `x - 3`, is `1`.

Therefore:

```
f'(x) = 2(x - 3) × 1
f'(x) = 2x - 6
```

This matters a lot for AI. A neural network is built in layers, where the output of one layer is the input to the next. That makes it a function inside a function inside a function, many times over. The chain rule is what lets us work out the derivative all the way through those layers.

## How does gradient descent work?

Gradient descent is all about finding the bottom of a function (where the gradient is 0) without knowing where the bottom is from the start. Its recipe has 4 steps:

1. Make a starting guess.
2. Work out the gradient at that point. If we were at the bottom, the gradient would be 0.
3. Take a small step in the opposite direction to the gradient. If the gradient is negative, step right. If the gradient is positive, step left.
4. Keep repeating steps 2 and 3 until the gradient is almost 0.

Step 3 uses this formula to work out the next x:

```
new_x = old_x - (learning_rate × gradient_at_old_x)
```

The minus sign is what makes us step the opposite way to the gradient. The learning rate is a small number (for example, `0.1`) which controls how big each step is.

The size of each step also depends on the gradient. When the ground is steep, we take big steps. As we get close to the bottom, the ground flattens out, so the gradient gets smaller and the steps get smaller too. This is why it settles at the bottom instead of running past it.

Choosing the learning rate matters:

- **Too big**, and each step jumps right over the bottom to the other side, so it keeps bouncing back and forth and never settles.
- **Too small**, and it gets there, but very slowly.

### Gradient descent example

- `f(x) = (x - 3)²`
- `f'(x) = 2x - 6`
- `learning_rate = 0.1`

The bottom of this curve is at `x = 3`. But let's pretend we didn't know that, and start our guess at `x = 1`.

```
f'(1) = 2(1) - 6 = -4
```

The gradient at `x = 1` is `-4`. It isn't 0, so we're not at the bottom yet. Let's find a new x:

```
new_x = 1 - (0.1 × -4)
new_x = 1.4
```

Then we repeat, and find the gradient at `x = 1.4`:

```
f'(1.4) = 2(1.4) - 6 = -3.2
```

The gradient is less steep than before, but still not 0, so we use the formula again. Here's how the first few steps go:

| Step | x | Gradient at x |
|---|---|---|
| 0 | 1 | -4 |
| 1 | 1.4 | -3.2 |
| 2 | 1.72 | -2.56 |
| 3 | 1.976 | -2.048 |
| 4 | 2.1808 | -1.6384 |
| 20 | 2.9769 | -0.0461 |

Notice, x gets closer and closer to 3, and the gradient gets closer and closer to 0. It never lands on exactly 3, but after enough steps it's so close that it doesn't matter.

### Finding the gradient without working out the derivative

In the example above, I worked out the derivative `f'(x) = 2x - 6` by hand. A real AI model is far too big to do that for. So my code estimates the gradient instead, by taking a tiny step and seeing how much the function changes:

```python
def numeric_slope(fn, x, h=0.0001):
    return (fn(x + h) - fn(x)) / h
```

This means my `gradient_descent` function works for any function I give it, without me working out a single derivative.

### With two inputs

A function like `f(x, y) = (x - 1)² + (y - 2)²` has two inputs, so there are two gradients: one for x and one for y. To find each one, you take a tiny step in that input only, while keeping the other one where it is. These are called partial derivatives, and putting them together in a list gives the gradient, for example `[8, 6]`.

The point becomes a list too, `[x, y]`, and the update is the same formula, just done with vectors:

```
new_point = old_point - (learning_rate × gradient)
```

Starting at `[5, 5]`, my code finds the bottom at `[1, 2]`.

## Why this is how AI models learn

An AI model is a big set of numbers, called weights. When you give the model an input, it uses its weights to make a guess. At the start the weights are random, so the guesses are bad.

To improve, we measure how wrong the guesses are with a single number called the error (also called the loss). The error depends on the weights: change the weights, and the error changes. So the error is a function of the weights, just like `f(x)` is a function of `x`.

For example, say a model guesses `y = w × x`, and for `x = 2` the right answer is `6`. Its error could be measured as:

```
error = (2w - 6)²
```

That's the same bowl shape as my examples, with its bottom at `w = 3`. Finding the best weight means finding the bottom of the error bowl, which is exactly what gradient descent does.

A real model works the same way, just much bigger. It has millions (or billions) of weights instead of one, so the gradient is a list of millions of numbers, one for each weight. Training the model means:

1. Make a guess and measure the error.
2. Work out the gradient of the error for every weight.
3. Nudge every weight a small step against its gradient, so the error goes down.
4. Repeat, millions of times.

That loop is what people mean when they say a model is "learning". My two-input code is the same idea, just with 2 numbers instead of millions.

One difference: my code estimates each gradient by nudging one input at a time. With millions of weights, that would mean running the model millions of extra times for every single step, which is far too slow. Real AI libraries use the chain rule instead (this is called backpropagation), which works out the gradient for every weight in one go, by going backwards through the layers. So the chain rule is what makes gradient descent fast enough to train real models.

## Running it

You need Python 3.9 or newer. There's nothing to install.

```
python gradient_descent.py
python gradient_descent_two_inputs.py
```