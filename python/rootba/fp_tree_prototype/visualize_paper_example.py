from fp_tree import FPTree

from networkx.drawing.nx_pydot import graphviz_layout
import pylab as plt
import networkx as nx

# datapoints from the paper
lms_obs = {
    1: [1, 2],
    2: [1, 2, 3],
    3: [1, 2, 3],
    4: [1, 2, 3],
    5: [1, 2, 3],
    6: [2, 3],
    7: [3, 4, 5],
    8: [5, 6],
    9: [5, 6],
    10: [5, 6]
}

# ========== Grow tree ==========
tree = FPTree(lms_obs)

# ========== Visualize ==========
edges, nodes = tree.get_edges()
G = nx.Graph()
for e in edges:
    G.add_edge(*e)

# draw cam idx
nodes_label = {node_hash: nodes[node_hash].cam_idx for node_hash in G.nodes()}
pos = graphviz_layout(G, prog="dot")
nx.draw(G, pos, labels=nodes_label, with_labels=True)

# draw lm idx
pos_bf_merge = {}
for node, coords in pos.items():
    pos[node] = (coords[0], coords[1] + 10)

# visualize result of merging
factors, non_factors, factors_map = tree.get_factors()
print(factors)
print(non_factors)

nx.draw_networkx_labels(G, pos, factors_map)
plt.show()
