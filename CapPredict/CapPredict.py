import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Generate a dataset of 30 S-curves
num_curves = 30
x = np.linspace(-10, 10, 500)  # Generate x values

# Initialize an empty list to store the curves
s_curves = []

for i in range(num_curves):
    # Adjust the steepness and midpoint of each S-curve randomly
    steepness = np.random.uniform(0.5, 2)
    midpoint = np.random.uniform(-5, 5)
    y = 1 / (1 + np.exp(-steepness * (x - midpoint)))  # Sigmoid function
    s_curves.append(y)

# Convert the curves into a DataFrame for analysis
s_curve_data = pd.DataFrame(s_curves).transpose()
s_curve_data.columns = [f"S_curve_{i+1}" for i in range(num_curves)]

# Plot a sample of the S-curves
plt.figure(figsize=(10, 6))
for i in range(5):  # Plot 5 random curves
    plt.plot(x, s_curves[i], label=f"S_curve_{i+1}")
plt.title("Sample of S-Curves")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
plt.show()