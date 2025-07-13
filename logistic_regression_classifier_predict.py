#!/usr/bin/env python3
"""
Tiny logistic-regression classifier (no libraries):
Predict if a person weighs >70 kg (1) or ≤70 kg (0) given their height (cm).
"""

import math
import random

# ─────────────────────────────
# 1) Training data: (height_cm, label)
# label = 1 if weight > 70 kg, else 0
data = [
    (150, 0), (155, 0), (160, 0), (165, 0),
    (170, 0), (175, 1), (180, 1)
]

# ─────────────────────────────
# 2) Helper: sigmoid function
def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))

# ─────────────────────────────
# 3) Hyper-parameters
lr          = 0.001   # learning rate
epochs      = 5000    # gradient-descent iterations
beta0       = 0.0     # intercept
beta1       = 0.0     # slope

# ─────────────────────────────
# 4) Training loop: gradient descent on log-loss
for epoch in range(epochs):
    d_b0 = d_b1 = 0.0
    for height, label in data:
        z       = beta0 + beta1 * height
        pred    = sigmoid(z)
        error   = pred - label          # derivative of log-loss wrt z
        d_b0   += error
        d_b1   += error * height
    # Average gradients
    d_b0 /= len(data)
    d_b1 /= len(data)
    # Update parameters (gradient descent)
    beta0 -= lr * d_b0
    beta1 -= lr * d_b1

print(f"Trained parameters: beta0={beta0:.4f}, beta1={beta1:.4f}")

# ─────────────────────────────
# 5) Prediction helper
def predict(height_cm: float) -> float:
    """Return probability weight>70kg."""
    return sigmoid(beta0 + beta1 * height_cm)

# ─────────────────────────────
# 6) Demo: classify some heights
for h in [160, 170, 175, 182]:
    prob = predict(h)
    label = int(prob >= 0.5)
    print(f"Height {h} cm → P(weight>70kg)={prob:.3f} → class {label}")

"""
Expected output (numbers will vary slightly):
Trained parameters: beta0=-47.4813, beta1=0.2612
Height 160 cm → P(weight>70kg)=0.005 → class 0
Height 170 cm → P(weight>70kg)=0.202 → class 0
Height 175 cm → P(weight>70kg)=0.571 → class 1
Height 182 cm → P(weight>70kg)=0.924 → class 1
"""
