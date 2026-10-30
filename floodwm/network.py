"""Road network + shelters. Synthetic lattice here; swap in OSM via load_osm_graph()."""
import networkx as nx
import numpy as np


def build_grid_roads(h, w, spacing, cell_size_m, z, n_shelters, rng):
    """Lattice of roads every `spacing` cells. Shelters sit on the highest ground."""
    g = nx.Graph()
    rows = list(range(spacing // 2, h, spacing))
    cols = list(range(spacing // 2, w, spacing))
    for r in rows:
        for c in cols:
            g.add_node((r, c), rc=(r, c), elev=float(z[r, c]))
    for r in rows:
        for c in cols:
            if (r, c + spacing) in g:
                g.add_edge((r, c), (r, c + spacing), cells=[(r, c + k) for k in range(spacing + 1)],
                           length_m=spacing * cell_size_m)
            if (r + spacing, c) in g:
                g.add_edge((r, c), (r + spacing, c), cells=[(r + k, c) for k in range(spacing + 1)],
                           length_m=spacing * cell_size_m)
    order = sorted(g.nodes, key=lambda n: -g.nodes[n]["elev"])
    shelters, min_sep = [], 3 * spacing
    for n in order:
        if all(abs(n[0] - s[0]) + abs(n[1] - s[1]) >= min_sep for s in shelters):
            shelters.append(n)
        if len(shelters) == n_shelters:
            break
    return g, shelters


def sample_origins(g, shelters, depth_final, n, rng, max_depth=0.15):
    """Residents: nodes that are not shelters (biased toward flood-prone nodes in reality)."""
    cand = [n_ for n_ in g.nodes if n_ not in shelters]
    idx = rng.choice(len(cand), size=min(n, len(cand)), replace=False)
    return [cand[i] for i in idx]


def load_osm_graph(place_name):  # pragma: no cover - requires osmnx + internet
    """Real-world road graph. Requires `pip install osmnx`. See docs/DATA_SOURCES.md."""
    import osmnx as ox
    g = ox.graph_from_place(place_name, network_type="walk")
    return nx.Graph(g)
