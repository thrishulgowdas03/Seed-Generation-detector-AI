"""
Skeletonizes the shoot mask into 1-pixel-wide centerlines, builds a
graph over the skeleton pixels, and provides a function to trace a
path from a given starting point until it hits a branch (tangle with
another shoot) or a clean tip.

This is what lets us reason about individual shoots even when many of
them visually overlap/cross in a dense, tangled paper-towel-roll photo.
"""

import numpy as np
import networkx as nx
from skimage.morphology import skeletonize


def build_skeleton_graph(shoot_mask_clean: np.ndarray):
    """
    shoot_mask_clean: binary mask (0/255) with small noise already removed.
    Returns: (skeleton_bool_array, graph, degrees_dict, endpoints_list)
    """
    skel = skeletonize(shoot_mask_clean > 0)
    ys, xs = np.where(skel)
    pixel_set = set(zip(xs.tolist(), ys.tolist()))

    G = nx.Graph()
    for x, y in pixel_set:
        G.add_node((x, y))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nb = (x + dx, y + dy)
                if nb in pixel_set:
                    G.add_edge((x, y), nb, weight=(dx ** 2 + dy ** 2) ** 0.5)

    degrees = dict(G.degree())
    endpoints = [n for n, d in degrees.items() if d == 1]
    return skel, G, degrees, endpoints


def trace_path(G: nx.Graph, degrees: dict, start_node: tuple, max_steps: int = 3000):
    """
    Walk from a skeleton endpoint through degree-2 pixels until hitting
    a branch point (degree >= 3, i.e. this shoot tangles with another)
    or another endpoint (a clean, unambiguous shoot with two free ends).

    Returns (path_length_px, termination_reason) where termination_reason
    is one of: 'branch', 'endpoint', 'dead_end', 'max_steps'.
    """
    prev = None
    curr = start_node
    length = 0.0
    steps = 0
    while steps < max_steps:
        if degrees[curr] != 2 and curr != start_node:
            reason = "branch" if degrees[curr] >= 3 else "endpoint"
            return length, reason
        neighbors = [n for n in G.neighbors(curr) if n != prev]
        if not neighbors:
            return length, "dead_end"
        nxt = neighbors[0]
        length += G[curr][nxt]["weight"]
        prev, curr = curr, nxt
        steps += 1
    return length, "max_steps"


def match_seeds_to_shoots(seeds: list, endpoints: list, G: nx.Graph, degrees: dict,
                           gap_multiplier: float = 2.0):
    """
    For each seed, finds the nearest skeleton endpoint (candidate shoot
    tip/base) and traces its path. gap_multiplier * seed_size sets how
    far away an endpoint can be and still count as "belonging" to this
    seed -- tune this if seeds are consistently mis-matched.

    Returns a list of dicts, one per seed, with:
        gap_to_endpoint, shoot_path_length, termination, shoot_found (bool)
    """
    results = []
    if not endpoints:
        return [{"shoot_found": False} for _ in seeds]

    endpoints_arr = np.array(endpoints)

    for seed in seeds:
        seed_size = max(seed["w"], seed["h"])
        gap_threshold = seed_size * gap_multiplier

        d = np.sqrt(
            (endpoints_arr[:, 0] - seed["cx"]) ** 2 +
            (endpoints_arr[:, 1] - seed["cy"]) ** 2
        )
        idx = np.argmin(d)
        gap = float(d[idx])

        if gap > gap_threshold:
            results.append({"shoot_found": False, "gap_to_endpoint": round(gap, 1)})
            continue

        ep = tuple(int(v) for v in endpoints_arr[idx])
        path_len, reason = trace_path(G, degrees, ep)
        results.append({
            "shoot_found": True,
            "gap_to_endpoint": round(gap, 1),
            "shoot_path_length": round(path_len, 1),
            "termination": reason,
        })

    return results
