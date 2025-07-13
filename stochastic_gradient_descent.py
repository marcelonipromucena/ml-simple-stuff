# SGD for fitting y = 2x + 1

import random

# Simple training data
data = [(1, 3), (2, 5), (3, 7), (4, 9)]  # y = 2x + 1

# Initialize weights
w, b = 0, 0
lr = 0.01

for epoch in range(100):
    x, y = random.choice(data)  # pick one example
    pred = w * x + b
    error = pred - y
    w -= lr * error * x
    b -= lr * error

print(f"Learned weights: w = {w:.2f}, b = {b:.2f}")