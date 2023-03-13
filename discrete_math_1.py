from copy import deepcopy
import networkx as nx

import matplotlib.pyplot as plt

graph = {
    "Armenia": ["Georgia", "Turkey"],
    "Albania": ["Greece", "North Macedonia", "Kosovo", "Montenegro"],
    "Andorra": ["France", "Spain"],
    "Austria": ["Germany", "Czech Republic", "Slovakia", "Hungary", "Slovenia", "Italy", "Switzerland",
                "Liechtenstein"],
    "Belarus": ["Russia", "Latvia", "Lithuania", "Poland", "Ukraine"],
    "Belgium": ["Netherlands", "Germany", "Luxembourg", "France"],
    "Bosnia and Herzegovina": ["Croatia", "Serbia", "Montenegro"],
    "Bulgaria": ["Romania", "Serbia", "North Macedonia", "Greece", "Turkey"],
    "Croatia": ["Slovenia", "Hungary", "Serbia", "Bosnia and Herzegovina", "Montenegro"],
    "Cyprus": [],
    "Czech Republic": ["Germany", "Poland", "Slovakia", "Austria"],
    "Denmark": ["Germany"],
    "Estonia": ["Russia", "Latvia"],
    "Finland": ["Sweden", "Norway", "Russia"],
    "France": ["Belgium", "Luxembourg", "Germany", "Switzerland", "Italy", "Spain", "Andorra", "Monaco"],
    "Germany": ["Denmark", "Poland", "Czech Republic", "Austria", "Switzerland", "France", "Luxembourg", "Netherlands",
                "Belgium"],
    'Georgia': ['Russia', 'Turkey', 'Armenia'],
    "Greece": ["Albania", "North Macedonia", "Bulgaria", "Turkey"],
    "Hungary": ["Austria", "Slovakia", "Ukraine", "Romania", "Serbia", "Croatia", "Slovenia"],
    "Iceland": [],
    "Ireland": ["United Kingdom"],
    "Italy": ["France", "Switzerland", "Austria", "Slovenia", "Vatican City", "San Marino"],
    "Kosovo": ["Albania", "Montenegro", "Serbia", "North Macedonia"],
    "Latvia": ["Estonia", "Russia", "Belarus", "Lithuania"],
    "Liechtenstein": ["Switzerland", "Austria"],
    "Lithuania": ["Latvia", "Belarus", "Poland", "Russia"],
    "Luxembourg": ["Germany", "Belgium", "France"],
    "Malta": [],
    "Moldova": ["Romania", "Ukraine"],
    "Monaco": ["France"],
    "Montenegro": ["Croatia", "Bosnia and Herzegovina", "Serbia", "Kosovo", "Albania"],
    "Netherlands": ["Germany", "Belgium"],
    "North Macedonia": ["Kosovo", "Serbia", "Albania", "Greece", "Bulgaria"],
    "Norway": ["Sweden", "Finland", "Russia"],
    "Poland": ["Germany", "Czech Republic", "Slovakia", "Ukraine", "Belarus", "Lithuania", "Russia"],
    "Portugal": ["Spain"],
    "Romania": ["Hungary", "Serbia", "Ukraine", "Moldova", "Bulgaria"],
    "Russia": ["Norway", "Finland", "Estonia", "Latvia", "Lithuania", "Poland", "Belarus", "Ukraine", "Georgia"],
    "San Marino": ["Italy"],
    "Serbia": ["Hungary", "Romania", "Bulgaria", "North Macedonia", "Kosovo", "Montenegro", "Bosnia and Herzegovina",
               "Croatia"],
    "Slovakia": ["Czech Republic", "Poland", "Ukraine", "Hungary", "Austria"],
    "Slovenia": ["Austria", "Hungary", "Croatia", "Italy"],
    "Spain": ["Portugal", "France", "Andorra"],
    "Sweden": ["Norway", "Finland"],
    "Switzerland": ["Germany", "Austria", "Liechtenstein", "Italy", "France"],
    "Turkey": ["Greece", "Bulgaria", "Georgia", "Armenia"],
    "Ukraine": ["Russia", "Belarus", "Poland", "Slovakia", "Hungary", "Romania", "Moldova"],
    "United Kingdom": ["Ireland"],
    "Vatican City": ["Italy"]
}

# Delete vertices not included into the largest connected component
graph.pop("Ireland")
graph.pop("United Kingdom")
graph.pop("Cyprus")
graph.pop("Malta")
graph.pop("Iceland")

# Uncomment for showing graph with matplotlib
# nx.draw_networkx(g)
# plt.show()

# Obviously (by definition)
edges = 0
delta_low = 1000000
delta_high = 0
for (key, lis) in graph.items():
    new_edges = len(lis)
    if new_edges:
        edges += new_edges
        delta_low = min(delta_low, new_edges)
        delta_high = max(delta_high, new_edges)
print("Edges:", edges / 2, "Delta_low:", delta_low, "Delta_High:", delta_high)


# Default BFS algorythm for finding maximum distance to another vertex in graph
def max_way_to_vert(vertex: str) -> int:
    vertices = [vertex]
    colored = {}
    for j in graph:
        colored[j] = -1
    colored[vertex] = 0
    while len(vertices):
        vert = vertices.pop(0)
        for s in graph[vert]:
            if colored[s] == -1:
                colored[s] = colored[vert] + 1
                vertices.append(s)
    dist = 0
    for j in colored:
        dist = max(dist, colored[j])
    return dist


# Radius and diameter are also found by definition
radius = 100000
diameter = 0
for i in graph:
    dst = max_way_to_vert(i)
    radius = min(radius, dst)
    diameter = max(diameter, dst)
print("Radius:", radius, "Diameter:", diameter)
print("Center:", end=" ")
center = []
for i in graph:
    if max_way_to_vert(i) == radius:
        center.append(i)
print(", ".join(center))
print("-----------------------")

# Checking previous results with networkx
g = nx.Graph(graph)
print("Rad:", nx.radius(g), "Diam:", nx.diameter(g),
      "\nCenter:", ", ".join(nx.center(g)))
print("Maximum clique:", ", ".join(nx.algorithms.approximation.max_clique(g)))
print("-----------------------")

# And finding other values with explanation of documentation
# Function finding maximum independent set abuses the Ramsey algorythm:
# Firstly, it splits graph on two parts:
# first part - current vertex and independent set of its neighbors,
# second part - independent set of vertices not connected to current vertex.
# Vertex is chosen accidentally.
# Next, result independent set is chosen as maximum of two sets.
# That's all.

max_stable_set = nx.algorithms.approximation.maximum_independent_set(g)
# or
# max_stable_set = nx.algorithms.approximation.ramsey_R2(g)[1]
print("Maximum stable set:", ", ".join(max_stable_set))
print("Set length:", len(max_stable_set))
print("-----------------------")

# Finding maximum matching
# We can write such greedy implementation, that takes edge
# and removes 2 vertex connected with it:

# def find_maximum_matching(G: nx.Graph, n: int):
#     mx = 0
#     for edge in G.edges():
#         gr = G.copy()
#         gr.remove_node(edge[0])
#         gr.remove_node(edge[1])
#         mx = max(mx, find_maximum_matching(gr, n))
#     return n + 1 + mx
#
# print(find_maximum_matching(g, 0))

# However, this code works with about 90 ^ 14 ticks on our graph,
# so it wouldn't be able to calculate result on python, but with c++ may be.
#
# networkx uses Blossom algorythm that is written
# in 700 strings of code. Its main idea is based on
# Berge's lemma, that says that matching M is maximum only if
# there is no M-augmenting path. So algorythm tries to augment
# current matching while we can do it.
# Augmenting algorythm consists of finding augment path or "blossom" - a part of graph
# that looks like a cycle. For this blossom we can find optimal configuration
# of matching. Asymptotics is O(E*V^2)

max_matching = nx.max_weight_matching(g)
print("Maximum matching:", end=" ")
for i in max_matching:
    print(i[0], "-", i[1], end=", ")
print()
print("Set length:", len(max_matching))
print("-----------------------")

# Minimum vertex covering
# Explained in Notion
removed_g = g.copy()
removed = ["Germany", "Denmark", "Luxembourg", "France",
           "Belgium", "Netherlands", "France", "Monaco",
           "Andorra", "Portugal", "Spain", "Switzerland",
           "Italy", "Austria", "Vatican City", "San Marino",
           "Liechtenstein"]
removed_g.remove_nodes_from(removed)

# Remove vertex and check if there are no remaining edges
# def find_minimum_vertex_cover(G: nx.Graph) -> int:
#     mn = 10000000
#     for vert in G:
#         gr = G.copy()
#         if not gr[vert]:
#             continue
#         gr.remove_node(vert)
#         if gr.edges():
#             mn = min(mn, find_minimum_vertex_cover(gr))
#         else:
#             return 1
#     return mn
#
#
# print(len(removed_g), removed_g.size())
# print(find_minimum_vertex_cover(removed_g) + 7)
# print("-----------------------")
# Algorythm is not fast yet, so I commented it

# Algorythm of finding minimum edge coloring
# Algorythm uses maximum matching with adding random edges connection
# for remaining vertices
min_edge_cover = nx.min_edge_cover(g)
# It uses unnecessary copies, I don't know why they did it.
cover_copy = min_edge_cover.copy()
for k in min_edge_cover:
    if (k[1], k[0]) in cover_copy:
        cover_copy.remove(k)
print("Minimum edge covering: ", end=" ")
for i in cover_copy:
    print(i[0], "-", i[1], end=", ")
print()
print("Set length:", len(cover_copy))
print("-----------------------")

# Weighted graph
weighted_g = {
    "Armenia": {"Georgia": 276, "Turkey": 1433},
    "Albania": {"Greece": 424, "North Macedonia": 173, "Kosovo": 72, "Montenegro": 114},
    "Andorra": {"France": 861, "Spain": 202},
    "Austria": {"Germany": 410, "Czech Republic": 299, "Slovakia": 88, "Hungary": 214, "Slovenia": 292,
                "Italy": 522, "Switzerland": 601, "Liechtenstein": 114},
    "Belarus": {"Russia": 929, "Latvia": 173, "Lithuania": 174, "Poland": 375, "Ukraine": 370},
    "Belgium": {"Netherlands": 211, "Germany": 675, "Luxembourg": 184, "France": 308},
    "Bosnia and Herzegovina": {"Croatia": 393, "Serbia": 260, "Montenegro": 198},
    "Bulgaria": {"Romania": 608, "Serbia": 318, "North Macedonia": 231, "Greece": 590, "Turkey": 937},
    "Croatia": {"Slovenia": 135, "Hungary": 369, "Serbia": 347, "Bosnia and Herzegovina": 393, "Montenegro": 163},
    "Cyprus": {},
    "Czech Republic": {"Germany": 355, "Poland": 237, "Slovakia": 174, "Austria": 299},
    "Denmark": {"Germany": 267},
    "Estonia": {"Russia": 691, "Latvia": 314},
    "Finland": {"Sweden": 397, "Norway": 829, "Russia": 1227},
    "France": {"Belgium": 308, "Luxembourg": 233, "Germany": 450, "Switzerland": 525, "Italy": 684, "Spain": 622,
               "Andorra": 861, "Monaco": 458},
    "Georgia": {"Russia": 1568, "Turkey": 1296, "Armenia": 276},
    "Germany": {"Denmark": 410, "Poland": 456, "Czech Republic": 355, "Austria": 410, "Switzerland": 522,
                "France": 450, "Luxembourg": 231, "Netherlands": 578, "Belgium": 675},
    "Greece": {"Albania": 424, "North Macedonia": 215, "Bulgaria": 590, "Turkey": 735},
    "Hungary": {"Austria": 214, "Slovakia": 88, "Ukraine": 343, "Romania": 454, "Serbia": 241, "Croatia": 369,
                "Slovenia": 346},
    "Iceland": {},
    "Ireland": {"United Kingdom": 311},
    "Italy": {"France": 684, "Switzerland": 601, "Austria": 522, "Slovenia": 426, "Vatican City": 0,
              "San Marino": 236},
    "Kosovo": {"Albania": 343, "Montenegro": 82, "Serbia": 309, "North Macedonia": 100},
    "Latvia": {"Estonia": 308, "Russia": 832, "Belarus": 211, "Lithuania": 265},
    "Liechtenstein": {"Switzerland": 40, "Austria": 26},
    "Lithuania": {"Latvia": 265, "Belarus": 169, "Poland": 382, "Russia": 717},
    "Luxembourg": {"Germany": 25, "Belgium": 30, "France": 187},
    "Malta": {},
    "Moldova": {"Romania": 372, "Ukraine": 518},
    "Monaco": {"France": 18},
    "Montenegro": {"Croatia": 95, "Bosnia and Herzegovina": 72, "Serbia": 91, "Kosovo": 82, "Albania": 179},
    "Netherlands": {"Germany": 575, "Belgium": 164},
    "North Macedonia": {"Kosovo": 100, "Serbia": 244, "Albania": 171, "Greece": 422, "Bulgaria": 176},
    "Norway": {"Sweden": 1619, "Finland": 1672, "Russia": 1606},
    "Poland": {"Germany": 651, "Czech Republic": 684, "Slovakia": 444, "Ukraine": 736, "Belarus": 199, "Lithuania": 382,
               "Russia": 1069},
    "Portugal": {"Spain": 629},
    "Romania": {"Hungary": 432, "Serbia": 476, "Ukraine": 531, "Moldova": 372, "Bulgaria": 602},
    "Russia": {"Norway": 1606, "Finland": 1313, "Estonia": 338, "Latvia": 717, "Lithuania": 832, "Poland": 1069,
               "Belarus": 543, "Ukraine": 1207, "Georgia": 1538},
    "San Marino": {"Italy": 14},
    "Serbia": {"Hungary": 352, "Romania": 476, "Bulgaria": 318, "North Macedonia": 244, "Kosovo": 309, "Montenegro": 91,
               "Bosnia and Herzegovina": 215, "Croatia": 383},
    "Slovakia": {"Czech Republic": 197, "Poland": 444, "Ukraine": 266, "Hungary": 168, "Austria": 74},
    "Slovenia": {"Austria": 230, "Hungary": 217, "Croatia": 101, "Italy": 214},
    "Spain": {"Portugal": 629, "France": 623, "Andorra": 158},
    "Sweden": {"Norway": 1619, "Finland": 395},
    "Switzerland": {"Germany": 743, "Austria": 454, "Liechtenstein": 10, "Italy": 332, "France": 437},
    "Turkey": {"Greece": 192, "Bulgaria": 223, "Georgia": 985, "Armenia": 729},
    "Ukraine": {"Russia": 1206, "Belarus": 620, "Poland": 526, "Slovakia": 753, "Hungary": 944, "Romania": 579,
                "Moldova": 135},
    "United Kingdom": {"Ireland": 463},
    "Vatican City": {"Italy": 0}
}
weighted_g.pop("Ireland")
weighted_g.pop("United Kingdom")
weighted_g.pop("Cyprus")
weighted_g.pop("Malta")
weighted_g.pop("Iceland")

# For future using...
copy_weighted = deepcopy(weighted_g)


# Finding minimum spanning tree for weighted graph
# with algorithm that we used on practice
# (when we always take minimal edge)
def find_min_edge_to_not_covered(covered):
    choose = ""
    parent = ""
    min_dst = 100000000
    for c in covered:
        for e in weighted_g[c]:
            if e not in covered and min_dst > weighted_g[c][e]:
                parent = c
                choose = e
                min_dst = weighted_g[c][e]
    del weighted_g[choose][parent]
    covered.add(choose)
    edges.append((parent, choose))
    return min_dst


covered_countries = {"Spain"}
edges = []
dst = 0

while len(covered_countries) != 44:
    dst += find_min_edge_to_not_covered(covered_countries)
# Check that tree is correct
print("Tree size:", len(edges))
print("Minimum spanning tree weight:", dst)
for i in edges:
    print(i[0], "--", i[1])
print("-----------------------")

weighted_g = deepcopy(copy_weighted)

tree = dict()
for i in weighted_g:
    for j in weighted_g[i]:
        if (i, j) in edges or (j, i) in edges:
            if i not in tree:
                tree[i] = {j: weighted_g[i][j]}
            else:
                tree[i][j] = weighted_g[i][j]
            if j not in tree:
                tree[j] = {i: weighted_g[j][i]}
            else:
                tree[j][i] = weighted_g[j][i]

weight = sum([sum([tree[i][j] for j in tree[i]]) for i in tree]) / 2
print("Spanning tree weight", weight)
print("-----------------")


# Showing tree
# gg = nx.Graph(tree)
# nx.draw_networkx(gg)
# plt.show()

# Finding centroid: find connected components, then finding
# max weighted of them with every vertex in tree
def find_max_component():
    max_components = []
    min_max_weighted = float("inf")

    # Finding components of connectivity
    def find_components(vertex, component):
        used[vertex] = 1
        components[vertex] = component
        for u in copy_graph[vertex]:
            if not used[u]:
                find_components(u, component)

    for vert in tree:
        copy_graph = deepcopy(tree)
        for s in copy_graph[vert]:
            del copy_graph[s][vert]
        del copy_graph[vert]

        comp = 0
        used = {d: 0 for d in copy_graph}
        components = {d: -1 for d in copy_graph}
        for vrt in copy_graph:
            if not used[vrt]:
                find_components(vrt, comp)
                comp += 1
        coms_weight = {c: 0 for c in set(list(components.values()))}
        for vrt in copy_graph:
            for t in copy_graph[vrt]:
                if components[vrt] == components[t]:
                    coms_weight[components[vrt]] += copy_graph[vrt][t]
        if min_max_weighted > max(coms_weight.values()):
            max_components = [vert]
            min_max_weighted = max(coms_weight.values())
        elif min_max_weighted == max(coms_weight.values()):
            max_components += [vert]
    return max_components


print("Centroid:", ", ".join(find_max_component()))
print("-----------------")

# Code for Prufer code
# Sorted tree
tree = {i: list(tree[i].keys()) for i in tree}
tree = {i: tree[i] for i in sorted(tree.keys())}
# print(tree)
numbers = {list(tree.keys())[i - 1]: i for i in range(1, len(tree.keys()) + 1)}
for i, name in enumerate(tree):
    print(i + 1, "-- {", ", ".join(list(map(lambda h: str(numbers[h]), tree[name]))), "}")
# Now we have data for graphviz
