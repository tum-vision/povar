from collections import defaultdict
from tqdm import tqdm


def load_bal(path):
    # BAL FORMAT:
    # <num_cameras> <num_points> <num_observations>
    # <camera_index_1> <point_index_1> <x_1> <y_1>

    # lm_idx -> [cam_idx_1, cam_idx_2, ...]

    lms_obs_dict = defaultdict(list)
    f = open(path, "r")
    num_cams, num_lms, num_obs = f.readline().split()
    num_cams, num_lms, num_obs = int(num_cams), int(num_lms), int(num_obs)

    lms_obs_dict = defaultdict(list)
    for line in f:
        l = line.split(" ")
        if len(l) == 1:
            break

        cam_idx, lm_idx = l[:2]
        cam_idx, lm_idx = int(cam_idx), int(lm_idx)
        lms_obs_dict[lm_idx].append(cam_idx)

    f.close()
    assert (len(lms_obs_dict) == num_lms)

    return lms_obs_dict, [num_cams, num_lms, num_obs]


class FPNode:

    def __init__(self, parent, level, cam_idx, idx):
        self.level = level
        self.cam_idx = cam_idx

        self.parent = parent
        self.childs = dict()  # use ordered map in C++
        self.lm_indices = []

        self.factor_idx = None
        # which factor possesses the lms
        self.possess_lms = None

        # unique identifer
        self.idx = idx

        # use to reduce the search space during merging
        self.factor_childs = []

        # self.mutex

    def append(self, value):
        # mutex lock
        self.lm_indices.append(value)

    def try_insert_child(self, cam_idx, idx):
        # mutex lock
        if cam_idx not in self.childs:
            self.childs[cam_idx] = FPNode(self, self.level + 1, cam_idx, idx)
            return True
        return False


class FPTree:
    num_nodes = 0
    max_level = 0  # for reserve in C++

    def __init__(self, lms_obs, verbose=True):
        self.root = FPNode(None, 0, -1, self.num_nodes)
        self.num_nodes += 1

        self.support = defaultdict(lambda: 0)
        self.non_empty_nodes = dict()  # nodes that have at least one lm
        self.verbose = verbose
        self.lms_obs = lms_obs
        self.lms_node = dict()

        # the following loops can be iterate concurrently
        for obs_indices in tqdm(self.lms_obs.values(),
                                total=len(self.lms_obs),
                                desc="Computing support",
                                disable=not self.verbose):
            for obs_idx in obs_indices:
                self.support[obs_idx] += 1

        for obs_indices in tqdm(self.lms_obs.values(),
                                total=len(self.lms_obs),
                                desc="Sorting observations",
                                disable=not self.verbose):
            obs_indices.sort(key=lambda x: (self.support[x], x))

        for lm_idx, obs_indices in tqdm(self.lms_obs.items(),
                                        total=len(self.lms_obs),
                                        desc="Inserting",
                                        disable=not self.verbose):
            self.insert(lm_idx, obs_indices, len(obs_indices) - 1)

    # parallel_for
    def insert(self, lm_idx, obs_indices, idx):
        # traverse
        it = self.root
        for idx in range(len(obs_indices) - 1, -1, -1):
            cam_idx = obs_indices[idx]
            if it.try_insert_child(cam_idx, self.num_nodes):
                self.num_nodes += 1

            it = it.childs[cam_idx]

        it.append(lm_idx)
        self.non_empty_nodes[it.idx] = it
        self.lms_node[lm_idx] = it

        self.max_level = max(self.max_level, it.level)

    # For visualizate small example only
    def get_edges(self):
        edges = []
        nodes = {self.root.idx: self.root}
        self.get_edges_(self.root, edges, nodes)
        return edges, nodes

    def get_edges_(self, node, edges, nodes):
        for next_node in node.childs.values():
            edges.append((node.idx, next_node.idx))
            nodes[next_node.idx] = next_node
            self.get_edges_(next_node, edges, nodes)

    def get_lm_indices(self, node_indices):
        lm_indices = []
        for nod_idx in node_indices:
            lm_indices.extend(self.non_empty_nodes[nod_idx].lm_indices)

        return lm_indices

    # Reference: Index Data Structure for Fast Subset and Superset Queries
    def get_superset(self, node, subset, subset_idx):
        # subset is a sorted list (based on support)
        if subset_idx < 0:
            return node.factor_idx

        it = subset[subset_idx]
        for cam_idx in node.factor_childs:
            child = node.childs[cam_idx]

            if self.support[cam_idx] >= self.support[it]:
                if cam_idx == it:
                    res = self.get_superset(child, subset, subset_idx - 1)
                else:
                    res = self.get_superset(child, subset, subset_idx)

                if res != None:
                    return res

    def get_factors(self):
        # ========== Compute and merge the factors from leaf to root ==========
        factors = dict()  # factor_idx -> factor
        non_factors = dict()  # factor_idx -> factor

        max_factor_level = 0

        # Use for reserve in C++
        num_non_factor_lms = 0

        cams_to_factor_indices = defaultdict(set)

        # Iterate all leaf nodes
        # NOTE:
        # leaf_node_idx is used as the idx of a factor
        # can't iterate concurrently, but it's already quite fast
        for leaf_node_idx, leaf_node in tqdm(self.non_empty_nodes.items(),
                                             total=len(self.non_empty_nodes),
                                             desc="Traversing",
                                             disable=not self.verbose):
            # NOTE: in the paper they start traverse also if len(factor_lms_idx) > node.level
            if not leaf_node.childs:
                num_lms = 0
                node_indices = []  # reserve max_level
                nodes = []

                # travese upward from the bottom
                it = leaf_node
                while it is not self.root:
                    # if the lms have't add to any factor yet
                    if not it.factor_idx:
                        nodes.append(it)
                        if it.lm_indices:
                            node_indices.append(it.idx)
                            num_lms += len(it.lm_indices)
                    it = it.parent

                # factor or non_factor
                if num_lms > leaf_node.level:
                    factors[leaf_node_idx] = node_indices

                    cam_indices = self.lms_obs[leaf_node.lm_indices[0]]
                    for cam_idx in cam_indices:
                        cams_to_factor_indices[cam_idx].add(leaf_node_idx)

                    for node in nodes:
                        # mark nodes belongs to this factor
                        node.factor_idx = leaf_node_idx
                        node.parent.factor_childs.append(node.cam_idx)

                        # remove lms from the non-factor that possess node
                        if node.possess_lms:
                            # or non_factors[node.possess_lms] = -1
                            non_factors[node.possess_lms].remove(node.idx)
                            node.possess_lms = leaf_node_idx  # this line is unnecessary

                    # skip the longer factor in the next merging streep
                    max_factor_level = max(max_factor_level, leaf_node.level)
                else:
                    non_factor_node_indices = set()
                    for node_idx in node_indices:
                        node = self.non_empty_nodes[node_idx]
                        if not node.possess_lms:
                            non_factor_node_indices.add(node_idx)
                            node.possess_lms = leaf_node_idx

                    non_factors[leaf_node_idx] = non_factor_node_indices
                    num_non_factor_lms += len(non_factor_node_indices)

        # ========== Merge the remaining factors ==========

        non_factor_nodes = []  # reserve len(non_factors)

        # For nodes that are still not factor, find a factor which has a camera superset
        # Add the node to that factor if found, else identitfy it as non factor
        # NOTE: can be iterate concurently
        for leaf_node_idx, node_indices in tqdm(non_factors.items(),
                                                total=len(non_factors),
                                                desc="Merging",
                                                disable=not self.verbose):
            leaf_node = self.non_empty_nodes[leaf_node_idx]

            factor_candidates = dict()
            factor_candidates[leaf_node_idx] = cams_to_factor_indices[leaf_node.cam_idx]

            it = leaf_node.parent
            while it is not self.root:
                cam_factors = cams_to_factor_indices[it.cam_idx]

                candidate_node_indices = list(factor_candidates.keys())  # copy
                for node_idx in candidate_node_indices:
                    factor_candidates[node_idx] = factor_candidates[node_idx].intersection(cam_factors)
                    if not factor_candidates[node_idx]:
                        non_factor_nodes.append(node_idx)
                        del factor_candidates[node_idx]

                if it.lm_indices and it.idx in node_indices:
                    factor_candidates[it.idx] = cam_factors

                it = it.parent

            if factor_candidates:
                for node_idx, factor_indices in factor_candidates.items():
                    factor_idx = next(iter(factor_indices))  # pick a random factor
                    factors[factor_idx].append(node_idx)

        final_factors = []  # reserve
        for factor_idx, node_indices in factors.items():
            lm_indices = self.get_lm_indices(node_indices)

            # A single factor: (cam_indices, lm_indices)s]
            obs_indices = self.lms_obs[self.non_empty_nodes[factor_idx].lm_indices[0]]
            final_factors.append((obs_indices, lm_indices))

        # reserve num_non_factor_lms
        non_factor_lms = self.get_lm_indices(non_factor_nodes)

        return final_factors, non_factor_lms, factors
