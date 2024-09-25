from collections import defaultdict
import numpy as np
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
from fp_tree import FPTree, load_bal

BAL_PATH = "../../../data/rootba/test/bal-ladybug-problem-49-7776-pre-shrink-1800.txt"

BLUE = np.array([86, 193, 255]) / 255.
RED = np.array([255, 0, 0]) / 255.
GREY = np.array([213, 213, 213]) / 255.

lms_obs_dict, nums = load_bal(BAL_PATH)
num_cams, num_lms, _ = nums

# Insert landmarks' observations into the tree
tree = FPTree(lms_obs_dict)
factors, non_factors, _ = tree.get_factors()

# Visualize large group first
factors.sort(key=lambda x: len(x[1]), reverse=True)

# Generate colors
cm = plt.cm.get_cmap("tab20")
colors = cm.colors * math.ceil(len(factors) / 20)

# Assign colors
row = 0
Jp = np.ones((num_lms, num_cams, 3))

# factors
for color, (_, lm_indices) in zip(colors, factors):
    # show the landmarks with the least observed camera first

    cams_count = defaultdict(lambda: 0)
    cam_to_lm_indices = defaultdict(set)
    for lm_idx in lm_indices:
        for cam in lms_obs_dict[lm_idx]:
            cams_count[cam] += 1
            cam_to_lm_indices[cam].add(lm_idx)

    remaining_lm_indices = set(lm_indices)
    sorted_cams = sorted(cam_to_lm_indices.keys(), key=lambda x: cam_to_lm_indices[x])

    for cam in sorted_cams:
        for lm_idx in cam_to_lm_indices[cam]:
            if lm_idx in remaining_lm_indices:
                lm_cams = lms_obs_dict[lm_idx]
                for lm_cam in lm_cams:
                    Jp[row:row + 1, lm_cam] = color[:3]
                row += 1
                remaining_lm_indices.remove(lm_idx)

# non factors
for lm_idx in non_factors:
    cams = lms_obs_dict[lm_idx]
    for cam in cams:
        Jp[row:row + 1, cam] = GREY
    row += 1


def disable_ticks(ax):
    for xlabel_i in ax.axes.get_xticklabels():
        xlabel_i.set_visible(False)
        xlabel_i.set_fontsize(0.0)
    for xlabel_i in ax.axes.get_yticklabels():
        xlabel_i.set_fontsize(0.0)
        xlabel_i.set_visible(False)
    for tick in ax.axes.get_xticklines():
        tick.set_visible(False)
    for tick in ax.axes.get_yticklines():
        tick.set_visible(False)


fig, ax = plt.subplots()
disable_ticks(ax)
ax.imshow(Jp, aspect=0.15)
plt.show()
