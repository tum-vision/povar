from fp_tree import FPTree, load_bal
import time

# BAL_PATH = "../../../data/rootba/test/bal-ladybug-problem-49-7776-pre-shrink-1800.txt"
BAL_PATH = "../../../data/rootba/bal/final/problem-93-61203-pre.txt"
# BAL_PATH = "/home/tin/Downloads/problem-394-100368-pre.txt"
# BAL_PATH = "/home/tin/Downloads/problem-1936-649673-pre.txt"
# BAL_PATH = "/home/tin/Downloads/problem-13682-4456117-pre.txt"

# ========== Load data ==========
lms_obs_dict, nums = load_bal(BAL_PATH)
_, num_lms, num_obs = nums

# ========== Grow tree ==========
tree = FPTree(lms_obs_dict)

# ========== Visualize ==========
st = time.time()
factors, non_factors, _ = tree.get_factors()
print("Time to group: {}".format(time.time() - st))

grouped_lms = set()
for f in factors:
    grouped_lms.update(f[1])

print("Groups:", len(factors))
print("Grouped lms:", len(grouped_lms))
print("Remaining lms: ", len(non_factors))

FACTORS_OBS_PATH = "./factors_obs.txt"
FACTORS_LMS_PATH = "./factors_lms.txt"
NON_FACTORS_LMS_PATH = "./non_factors_lms.txt"

with open(FACTORS_OBS_PATH, "a") as f:
    for factor_obs, _ in factors:
        f.write(" ".join(map(str, factor_obs)) + "\n")

with open(FACTORS_LMS_PATH, "a") as f:
    for _, factor_lms in factors:
        f.write(" ".join(map(str, factor_lms)) + "\n")

with open(NON_FACTORS_LMS_PATH, "a") as f:
    f.write(" ".join(map(str, non_factors)) + "\n")
