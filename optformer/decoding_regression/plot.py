import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

shape_name = "hollow_square" # hollow_square/zigzag/spiral/half_moons
tokenizer = "stern_brocot"

ground_truth = pd.read_csv(f"/home/sankalp/optformer/optformer/decoding_regression/data/ground_truth_{tokenizer}_{shape_name}.csv")
predicted = pd.read_csv(f"/home/sankalp/optformer/optformer/decoding_regression/data/predicted_{tokenizer}_{shape_name}.csv")

fig, axs = plt.subplots(1, 2, figsize=(6, 3))

axs[0].scatter(ground_truth["x"], ground_truth["y"], alpha=0.1)
# axs[0].set_xlim(-0.05, 1.05)
# axs[0].set_aspect('auto')
# axs[0].set_ylim(-1, 1.5)
axs[0].set_title("Ground Truth")

axs[1].scatter(predicted["x"], predicted["y"], alpha=0.1)
axs[1].set_xlim(-1.2983806381155862, 1.27493284219296)
# axs[1].set_aspect('auto')
axs[1].set_ylim(-1.276318250430117, 1.281084622569874)
axs[1].set_title("Sampled")
# plt.scatter(points[:, 0], points[:, 1], alpha=0.5, label='Sampled')
# plt.legend()
plt.tight_layout()
# plt.title("Density Estimation via Autoregressive Decoder")
plt.savefig(f"/home/sankalp/optformer/optformer/decoding_regression/density_plots/{tokenizer}/{shape_name}", dpi = 400)

print("Predicted Limits")
print("X limits:", axs[1].get_xlim())
print("Y limits:", axs[1].get_ylim())

print("Ground Truth Limits")
print("X limits:", axs[0].get_xlim())
print("Y limits:", axs[0].get_ylim())