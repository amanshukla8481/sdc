import numpy as np
import math
def find_desired_angle(center_line_coeffs, y_h, x_w):
    x_car = x_w // 2 
    y_car = y_h - 1 
    # car_pos = (x_car, y_car) 
    y_dest = y_h // 2 + 80
    x_dest = np.polyval(center_line_coeffs, y_dest) 
    dest_pos = (int(x_dest), int(y_dest)) 

    # To find Angle 
    # Define point C (same x as car, y = y_dest) 
    c_pos = (x_car, int(y_dest)) 

    # Vectors from A (car) to B and A to C 
    AB = (dest_pos[0] - x_car, dest_pos[1] - y_car)   # (dx_AB, dy_AB) 
    AC = (c_pos[0] - x_car, c_pos[1] - y_car)         # (dx_AC, dy_AC) -> will be (0, y_dest - y_car) 

    # Convert to numpy for convenience 
    AB_v = np.array(AB, dtype=float) 
    AC_v = np.array(AC, dtype=float) 

    # # --- numeric (unsigned) angle using dot product, safe clamping --- 
    # dot = float(np.dot(AB_v, AC_v)) 
    # norms = (np.linalg.norm(AB_v) * np.linalg.norm(AC_v)) 
    # if norms == 0: 
    #     angle_deg = 0.0 
    #     # print(angle_deg) 
    # else: 
    #     cosang = dot / norms 
    #     # clamp to valid range to avoid numeric errors 
    #     cosang = max(-1.0, min(1.0, cosang)) 
    #     angle_rad = math.acos(cosang) 
    #     angle_deg = math.degrees(angle_rad) 
        # print(angle_deg) 
    # --- signed angle (optional) using atan2: helpful for left/right sign --- 
    # Note: image y increases downward. We invert y for Cartesian angle computation. 
    def atan2_cart(v): 
        return math.atan2(-v[1], v[0])  # '-v[1]' converts image coordinates to Cartesian 

    ang_AB = atan2_cart(AB_v) 
    ang_AC = atan2_cart(AC_v) 
    signed_angle_rad = (ang_AB - ang_AC) 
    # normalise into [-pi, pi] 
    signed_angle_rad = (signed_angle_rad + math.pi) % (2*math.pi) - math.pi 
    signed_angle_deg = math.degrees(signed_angle_rad)

    return signed_angle_deg, ang_AB, ang_AC