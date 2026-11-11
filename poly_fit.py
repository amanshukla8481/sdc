import numpy as np
def poly_fit_lane(lane_points, degree=2):
    if len(lane_points) < degree + 1:
        return None

    x_coords = np.array([pt[1] for pt in lane_points])  # column indices
    y_coords = np.array([pt[0] for pt in lane_points])  # row indices

    # Fit polynomial
    coeffs = np.polyfit(y_coords, x_coords, degree)  # x = f(y)

    return coeffs
    