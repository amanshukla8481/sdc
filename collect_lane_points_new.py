import math

def collect_lane_points(mask, start_r, start_c,
                                       radius=10,
                                       desired_angle_deg=60.0,
                                       angle_tol_deg=20.0,
                                       max_iters=100):
    """
    mask: 2D numpy array (values 0 or 255)
    start_r, start_c: starting pixel (row, col)
    radius: search radius (pixels)
    desired_angle_deg: desired direction in degrees (see convention below)
    angle_tol_deg: +/- tolerance around desired_angle_deg
    max_iters: maximum iterations for frontier expansion

    Returns: list of (r, c) collected points
    ------------------------------------------------
    """

    rows, cols = mask.shape
    # safety check
    if not (0 <= start_r < rows and 0 <= start_c < cols):
        return []

    if mask[start_r, start_c] != 255:
        # start not on lane
        return []

    visited = set()
    collected = set()
    frontier = {(start_r, start_c)}
    visited.add((start_r, start_c))
    collected.add((start_r, start_c))

    # precompute radius squared for distance test
    r2_limit = radius * radius
    desired = float(desired_angle_deg)
    tol = float(angle_tol_deg)

    for it in range(max_iters):
        # for each point in frontier, compute the set of candidate points that:
        #  - are mask==255
        #  - within radius
        #  - angle within tolerance of desired
        candidate_sets = []
        for (r, c) in frontier:
            S = set()
            rmin = max(0, r - radius)
            rmax = min(rows - 1, r + radius)
            cmin = max(0, c - radius)
            cmax = min(cols - 1, c + radius)

            # scan bounding box (could be optimized with numpy if needed)
            for rr in range(rmin, rmax + 1):
                dy_pix = r - rr                 # signed vertical (image y flipped)
                dy2 = (rr - r) * (rr - r)      # for distance check (positive)
                for cc in range(cmin, cmax + 1):
                    if mask[rr, cc] != 255:
                        continue

                    dx = cc - c
                    dist2 = dx * dx + dy2
                    if dist2 > r2_limit:
                        continue

                    # compute angle of vector from (r,c) to (rr,cc)
                    # dy = r - rr (so upward vectors give positive dy)
                    ang = math.degrees(math.atan2(dy_pix, dx))
                    # normalize difference to [-180, 180]
                    diff = (ang - desired + 180.0) % 360.0 - 180.0
                    if abs(diff) <= tol:
                        # candidate passes
                        if (rr, cc) not in visited:
                            S.add((rr, cc))

            candidate_sets.append(S)

        if not candidate_sets:
            break

        # intersection of all sets -> common pixels acceptable to all frontier points
        # If any set empty, intersection becomes empty -> termination
        next_frontier = set.intersection(*candidate_sets)

        # remove already visited ones (safety)
        next_frontier -= visited

        if not next_frontier:
            break

        # update visited and collected, prepare next iteration
        visited.update(next_frontier)
        collected.update(next_frontier)
        frontier = next_frontier

    return list(collected)
