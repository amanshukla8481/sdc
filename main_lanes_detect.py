import numpy as np
def extract_main_lanes(ll_seg_mask):
    ll_seg_mask *= 255 
    y_h, x_w = ll_seg_mask.shape 
    # Find starting points for DFS 
    lane_points = [] 
    count = 0 
    y_line = y_h // 2 + 50  # Scan line 
        
    for i in range(0, x_w): 
        value = ll_seg_mask[y_line][i] 
        if value == 255: 
            count += 1 
        else: 
            if count >= 5: # Found a potential lane start 
                lane_points.append(i - 2) 
            count = 0 
        
    # Find the two main lanes (widest gap) 
    distance = 0 
    left_lane = 0 
    right_lane = 0 
    for i in range(len(lane_points) - 1): 
        if lane_points[i+1] - lane_points[i] > distance: 
            distance = lane_points[i+1] - lane_points[i] 
            left_lane = lane_points[i] 
            right_lane = lane_points[i+1] 

    lane_points = [left_lane, right_lane] # Keep only the two main lanes
    return lane_points 
            