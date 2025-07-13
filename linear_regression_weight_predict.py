#!/usr/bin/env python3
"""
Simple linear-regression demo:
Predict a person's weight (kg) from their height (cm) using ordinary
least squares — implemented with only the Python standard library.
"""

# ─────────────────────────────────────────────────────────────
# 1. Toy training data  (height_cm, weight_kg)
#    Feel free to add / adjust!
data = [
    (150, 50),
    (155, 52),
    (160, 56),
    (165, 59),
    (170, 63),
    (175, 68),
    (180, 72),
]

# ─────────────────────────────────────────────────────────────
# 2. Compute means of x (height) and y (weight)
n = len(data)
x_mean = sum(h for h, _ in data) / n
y_mean = sum(w for _, w in data) / n

# 3. Calculate numerator & denominator for β1 (slope)
num = 0.0  # Σ(x−x̄)(y−ȳ)
den = 0.0  # Σ(x−x̄)²
for height, weight in data:
    dx = height - x_mean
    dy = weight - y_mean
    num += dx * dy
    den += dx * dx

beta1 = num / den                 # slope
beta0 = y_mean - beta1 * x_mean   # intercept

print("Fitted equation:")
print(f"    weight̂ = {beta0:.2f}  +  {beta1:.4f} × height")

# ─────────────────────────────────────────────────────────────
# 4. Little REPL to predict new weights
def predict(height_cm: float) -> float:
    """Return predicted weight (kg) for a given height (cm)."""
    return beta0 + beta1 * height_cm

print("\nType a height in cm (or just press Enter to quit)")
while True:
    try:
        raw = input("Height (cm): ").strip()
        if not raw:
            break
        h_cm = float(raw)
        w_pred = predict(h_cm)
        print(f"Predicted weight ≈ {w_pred:.1f} kg\n")
    except ValueError:
        print("Please enter a numeric value!\n")

print("Bye! 🙂")