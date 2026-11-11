import cv2
import numpy as np 
def enhance_image_for_sunlight_glare(frame): 
    # --- Preprocessing for bright sunlight and glare --- 
    # 1. Convert to LAB color space to separate brightness from color 
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB) 
    l, a, b = cv2.split(lab) 

    # 2. Adaptive histogram equalization on L-channel 
    #    This enhances local contrast without blowing up highlights 
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) 
    cl = clahe.apply(l) 

    # 3. Recombine and convert back to BGR 
    limg = cv2.merge((cl, a, b)) 
    frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR) 

    # 4. Apply adaptive gamma correction (reduces glare from sunlight) 
    #    Compute mean intensity and dynamically tune gamma 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    mean_intensity = np.mean(gray) 
    gamma = 1.5 if mean_intensity > 180 else 1.0 if mean_intensity > 120 else 0.8 

    frame = np.clip(((frame / 255.0) ** (1.0 / gamma)) * 255, 0, 255).astype(np.uint8) 

    # 5. Optional: shadow & highlight normalization 
    #    Balances dark and bright zones for road surfaces 
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB) 
    l, a, b = cv2.split(lab) 
    l = cv2.normalize(l, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX) 
    frame = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    return frame