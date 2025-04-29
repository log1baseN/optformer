import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ground_truth = pd.read_csv("/home/sankalp/optformer/optformer/decoding_regression/data/ground_truth.csv")
predicted = pd.read_csv("/home/sankalp/optformer/optformer/decoding_regression/data/predicted.csv")

fig, axs = plt.subplots(1, 2, figsize=(6, 3))

axs[0].scatter(ground_truth["x"], ground_truth["y"], alpha=0.1)
# axs[0].set_xlim(-0.05, 1.05)
# axs[0].set_aspect('auto')
# axs[0].set_ylim(-1, 1.5)
axs[0].set_title("Ground Truth")

axs[1].scatter(predicted["x"], predicted["y"], alpha=0.1)
axs[1].set_xlim(-1, 1.05)
# axs[1].set_aspect('auto')
axs[1].set_ylim(-1.2, 1.2)
axs[1].set_title("Sampled")
# plt.scatter(points[:, 0], points[:, 1], alpha=0.5, label='Sampled')
# plt.legend()
plt.tight_layout()
plt.title("Density Estimation via Autoregressive Decoder")
plt.savefig("/home/sankalp/density_estimation_spiral.png", dpi = 300)

print("X limits:", axs[1].get_xlim())
print("Y limits:", axs[1].get_ylim())