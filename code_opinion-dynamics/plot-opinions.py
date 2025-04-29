import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

output_folder = 'figures/opinions-only'
os.makedirs(output_folder, exist_ok=True)

distance_with_interact = mean_lf = np.loadtxt('figures/opinions-only/opinion-distance_with-interactions.txt')
distance_no_interact = mean_lf = np.loadtxt('figures/opinions-only/opinion-distance_no-interactions.txt')

steps = len(distance_no_interact)

fig, ax = plt.subplots(figsize=(5, 5))
ax.set_title("Opinion distance", fontweight="bold", fontsize=16)
ax.set_xlabel("Time step")
ax.set_ylabel("Distance")
ax.set_xlim(1, steps)
ax.set_ylim(-0.1, 0.1)

ax.plot(range(1, steps+1), distance_no_interact, label="No interaction", linewidth=2)
ax.plot(range(1, steps+1), distance_with_interact, label="With interaction", linewidth=2)

ax.legend(loc='lower left')
plt.tight_layout()

output_file = os.path.join(output_folder, f"opinion-distance.svg")
plt.savefig(output_file)