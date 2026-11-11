def collect_lane_points(mask, start_r, start_c):
    visited = set()
    # def dfs(mask, start_r, start_c, visited): 
    stack = [(start_r, start_c)] 
    collected_points = [] 
    while stack: 
        r, c = stack.pop() 
        if (r, c) in visited: 
            continue 
        if r < 0 or c < 0 or r >= mask.shape[0] or c >= mask.shape[1]: 
            continue 
        if mask[r, c] != 255: 
            continue 

        visited.add((r, c)) 
        collected_points.append((r, c)) 

        # move up, left, right 
        stack.append((r - 1, c))     # up 
        stack.append((r, c - 1))     # left 
        stack.append((r, c + 1))     # right 
    visited.remove((start_r, start_c)) 
    visited.remove((start_r, start_c-1)) 
    visited.remove((start_r, start_c+1)) 
    stack.append((start_r, start_c)) 
    while stack: 
        r, c = stack.pop() 
        if (r, c) in visited: 
            continue 
        if r < 0 or c < 0 or r >= mask.shape[0] or c >= mask.shape[1]: 
            continue 
        if mask[r, c] != 255: 
            continue 

        visited.add((r, c)) 
        collected_points.append((r, c)) 

        # move down, left, right 
        stack.append((r + 1, c))     # down 
        stack.append((r, c - 1))     # left 
        stack.append((r, c + 1))     # right 
    return collected_points 