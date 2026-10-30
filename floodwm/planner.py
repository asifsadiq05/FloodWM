"""Time-dependent evacuation routing over a (predicted) flood depth timeline."""
import heapq


def edge_depth(depth_t, cells):
    return max(float(depth_t[r, c]) for r, c in cells)


def plan_route(g, origin, shelters, depth_tl, depart_step, dt_s, speed_mps,
               max_safe_depth, risk_weight=0.0, frozen=False):
    """Earliest-arrival Dijkstra. depth_tl: [T+1,H,W] depth timeline (metres).

    An edge is usable only if, at the time the evacuee enters it, the maximum depth
    along it is below `max_safe_depth`. Speed degrades with depth. frozen=True uses
    the depth at departure for the whole trip (a 'reactive / no forecast' baseline);
    frozen=None-depth (all zeros) reproduces a static shortest path.
    Returns (path, arrival_seconds) or (None, inf).
    """
    T = depth_tl.shape[0] - 1
    t0 = depart_step * dt_s
    best, prev, pq = {origin: t0}, {}, [(t0, origin)]
    targets = set(shelters)
    while pq:
        t, u = heapq.heappop(pq)
        if t > best.get(u, float("inf")):
            continue
        if u in targets:
            path = [u]
            while path[-1] in prev:
                path.append(prev[path[-1]])
            return path[::-1], t
        k = min(int(t // dt_s), T)
        k = depart_step if frozen else k
        for v in g.neighbors(u):
            e = g[u][v]
            d = edge_depth(depth_tl[k], e["cells"])
            if d >= max_safe_depth:
                continue
            speed = speed_mps * (1.0 - 0.8 * d / max_safe_depth)
            nt = t + e["length_m"] / speed * (1.0 + risk_weight * d)
            if nt < best.get(v, float("inf")):
                best[v], prev[v] = nt, u
                heapq.heappush(pq, (nt, v))
    return None, float("inf")


def execute_route(g, path, true_depth_tl, depart_step, dt_s, speed_mps, max_safe_depth):
    """Walk a planned path through the TRUE flood. Returns (reached, trapped_at_edge)."""
    if path is None:
        return False, None
    T = true_depth_tl.shape[0] - 1
    t = depart_step * dt_s
    for u, v in zip(path[:-1], path[1:]):
        e = g[u][v]
        d = edge_depth(true_depth_tl[min(int(t // dt_s), T)], e["cells"])
        if d >= max_safe_depth:
            return False, (u, v)
        t += e["length_m"] / (speed_mps * (1.0 - 0.8 * d / max_safe_depth))
    return True, None
