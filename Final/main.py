import time 
import cv2 
from collect_lane_points import collect_lane_points
from find_desired_angle import find_desired_angle
from main_lanes_detect import extract_main_lanes
from poly_fit import poly_fit_lane
import torch 
import numpy as np 
import math 
from utils.utils import ( 
    select_device, time_synchronized, 
    split_for_trace_model, lane_line_mask, 
    driving_area_mask, show_seg_result, 
    non_max_suppression, scale_coords, 
    plot_one_box, letterbox 
) 
import threading 
import serial 
import time 
import image_enhancement as ie

import serial
import time
import threading

class SerialManager:
    def __init__(self):
        self.ser = None
        self.lock = threading.Lock() # To make sending data thread-safe

    def connect(self, port, baudrate, timeout=1):
        try:
            self.ser = serial.Serial(port, baudrate, timeout=timeout)
            print(f"Waiting for serial connection on {port}...")
            time.sleep(2) # Allow time for the connection to establish
            print(f"Successfully connected to {port}.")
            return True
        except serial.SerialException as e:
            print(f"[ERROR] Could not connect to {port}: {e}")
            self.ser = None
            return False

    def send(self, data_str):
        if not self.ser or not self.ser.is_open:
            print("[ERROR] Cannot send: Serial port is not connected.")
            return
        with self.lock:
            try:
                self.ser.write(data_str.encode())
            except serial.SerialException as e:
                print(f"[ERROR] Failed to send data: {e}")

    def receive(self):
        if not self.ser or not self.ser.is_open:
            print("[ERROR] Cannot receive: Serial port is not connected.")
            return None
        try:
            line = self.ser.readline().decode('utf-8', errors='ignore').strip()
            return line
        except serial.SerialException as e:
            print(f"[ERROR] Failed to read data: {e}")
            return None

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("Serial connection closed.")
 
 
def detect_yolopv2_camera(weights='data/weights/yolopv2.pt', img_size=640, camera_id=0, conf_thres=0.3, iou_thres=0.45): 
    # --- Setup model --- 
    device = select_device('')  # automatically selects CUDA if available 
    model = torch.jit.load(weights, map_location=device) 
    model.to(device).eval() 
 
    half = device.type != 'cpu'  # use FP16 if on GPU 
 
    if half: 
        model.half() 

    arduino = SerialManager()
    if not arduino.connect('/dev/ttyACM0', 115200):
        print("[WARNING] Running without Arduino connection.")
 
    # --- Initialize camera --- 
    cap = cv2.VideoCapture(camera_id) 
    if not cap.isOpened(): 
        print("[ERROR] Cannot open camera.") 
        return 
 
    print(f"\n[INFO] Starting live camera feed on device: {device}\nPress 'q' to quit.\n") 
 
    frame_id=0 
    prev_center_line_coeffs = None
    MAX_CENTER_SHIFT = 100  # pixels; threshold for rejecting sudden jumps
    SMOOTHING_ALPHA = 0.2   # how much new data affects the smoothed output
    h,w=720,1280 #height and width of the frame
    
    while True: 
        ret, frame = cap.read() 
         
        if not ret: 
            print("[INFO] Camera feed ended or cannot read frame.") 
            break 

        #Image Enhancingement for Sunlight Glare
        frame = ie.enhance_image_for_sunlight_glare(frame)
        
        frame_id+=1 
        im0 = cv2.resize(frame, (1280, 720))
        im0[:h//2-120, :] = 0 
        img = letterbox(im0, img_size, stride=32)[0] 
 
        # BGR → RGB and HWC → CHW 
        img = img[:, :, ::-1].transpose(2, 0, 1) 
        img = np.ascontiguousarray(img) 
 
        img = torch.from_numpy(img).to(device) 
        img = img.half() if half else img.float()  # FP16 for GPU, FP32 for CPU 
        img /= 255.0 
        if img.ndimension() == 3: 
            img = img.unsqueeze(0) 
 
        # --- Inference --- 
        with torch.no_grad(): 
            t1 = time_synchronized() 
            [pred, anchor_grid], seg, ll = model(img) 
            t2 = time_synchronized() 
 
        # --- Segmentation masks --- 
        ll_seg_mask = lane_line_mask(ll) 
        da_seg_mask = np.zeros_like(ll_seg_mask)
 
        show_seg_result(im0, (da_seg_mask, ll_seg_mask), is_demo=True)  
        
        # --- Extract main lanes --- 
        lane_points = extract_main_lanes(ll_seg_mask)

        # --- Collect pixels and fit polynomials ---

        visited = set() 
        lanes_data = [] 
        y_h, x_w = ll_seg_mask.shape
        y_line = y_h // 2 + 50
         
        for x in lane_points: 
            if x > 0 and ll_seg_mask[y_line, x] == 255 and (y_line, x) not in visited: 
                lane_pixels = collect_lane_points(ll_seg_mask, y_line, x) 
                lanes_data.append(lane_pixels) 
 
        poly_fits = []  # store polynomial coefficients 
         
        for idx, lane in enumerate(lanes_data): 
            lane = np.array(lane)
            if len(lane) < 20:  # not enough points to fit 
                continue 
            coeffs = poly_fit_lane(lane, degree=2)
            poly_fits.append(coeffs) 
 
        # --- Calculate Center Line --- 
        # This is the "more accurate way" you asked for. 
        center_line_coeffs = None

        if len(poly_fits) == 2:
            coeffs_left = poly_fits[0]
            coeffs_right = poly_fits[1]
            center_line_coeffs = (coeffs_left + coeffs_right) / 2
            prev_center_line_coeffs = center_line_coeffs  # ✅ save for next frame
            poly_fits.append(center_line_coeffs)

        elif len(poly_fits) == 3:
            center_line_coeffs = poly_fits[2]
            prev_center_line_coeffs = center_line_coeffs  # ✅ save for next frame

        # ✅ If no valid center line this frame, reuse last one
        elif prev_center_line_coeffs is not None:
            center_line_coeffs = prev_center_line_coeffs
            poly_fits.append(center_line_coeffs)
             
        # --- Visualization of fitted lanes --- 
        colors = [(0, 255, 0), (0, 255, 0), (255, 0, 0), (255, 255, 0)]  # R, G, B... 
         
        for idx, coeffs in enumerate(poly_fits): 
            plot_y = np.linspace(y_h // 3, y_h - 1, num=100)
            x_vals_fit = np.polyval(coeffs, plot_y) 
            points = np.array([np.column_stack((x_vals_fit, plot_y))], dtype=np.int32) 
            color = colors[idx % len(colors)] 
            cv2.polylines(im0, points, isClosed=False, color=color, thickness=3) 
 
        # --- THIS IS THE NEW LOGIC YOU ASKED FOR --- 
        if center_line_coeffs is not None:
            # 1. Define car position A
            x_car = x_w // 2 
            y_car = y_h - 1 
            car_pos = (x_car, y_car) 
            y_dest = y_h // 2 + 80
            x_dest = np.polyval(center_line_coeffs, y_dest) 
            dest_pos = (int(x_dest), int(y_dest))
            c_pos = (x_car, int(y_dest)) #dest point C

            signed_angle_deg, ang_AB, ang_AC = find_desired_angle(center_line_coeffs, y_h, x_w)
 
            # --- Draw results on im0 --- 
            # Text: show unsigned angle and signed angle (signed sign indicates rotation direction) 
            signed_text = f"signed: {signed_angle_deg:+.1f} deg" 
            cv2.putText(im0, signed_text, (x_car - 150, y_car - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA) 
 
            # Draw small arc to visualise angle (centered at A) 
            # Compute start/end angles in degrees for drawing. Convert radians->degrees and normalize. 
            start_deg = math.degrees(ang_AC)  # angle of AC 
            end_deg   = math.degrees(ang_AB)  # angle of AB 
 
            start_cv = (start_deg) % 360 
            end_cv   = (end_deg) % 360 
 
            # draw arc (ellipse) - tune axes size if needed 
            arc_center = (x_car, y_car) 
            axes = (40, 40) 
            cv2.ellipse(im0, arc_center, axes, 0, start_cv, end_cv, (0, 255, 255), 2, cv2.LINE_AA) 
 
            # Draw small helper lines from A to B and A to C 
            cv2.line(im0, car_pos, dest_pos, (255, 0, 255), 2, cv2.LINE_AA)  # AB 
            cv2.line(im0, car_pos, c_pos, (0, 255, 0), 2, cv2.LINE_AA)      # AC 
            cv2.circle(im0, car_pos, 5, (0, 0, 255), -1)                    # A 
            cv2.circle(im0, dest_pos, 4, (255, 0, 255), -1)                # B 
            cv2.circle(im0, c_pos, 4, (0, 255, 0), -1)                     # C 
 
            #Check for steering rotation 
            color = (255, 0, 0)
            if abs(signed_angle_deg) > 25:
                signed_angle_deg = signed_angle_deg*2
            signed_angle_deg = np.clip(signed_angle_deg, -70, 70)

            angle = str(int(signed_angle_deg))+"\n"
            thread1 = threading.Thread(target=arduino.send, args=(angle,)) # Pass the string
            thread1.start() 
            # thread1.join() 

            thread2 = threading.Thread(target=arduino.receive)
            thread2.start() 
            # thread2.join()
 
            # 4. Draw the "optimal path" line (the steering correction) 
            # (Color: Magenta) 
            cv2.line(im0, car_pos, dest_pos, (255, 0, 255), 3, cv2.LINE_AA) 
             
            # 5. Draw the destination point as a circle 
            # (Color: Magenta) 
            cv2.circle(im0, dest_pos, 10, (255, 0, 255), -1) 
            cv2.circle(im0, car_pos, 10, (255, 0, 255), -1) 
 
        # --- Display --- 
        fps = 1 / (t2 - t1) 
        cv2.putText(im0, f'FPS: {fps:.1f}', (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2) 
        cv2.namedWindow("YOLOPv2 - Camera (CUDA)", cv2.WINDOW_NORMAL)  # make window resizable
        cv2.resizeWindow("YOLOPv2 - Camera (CUDA)", 640, 480) 
        cv2.imshow("YOLOPv2 - Camera (CUDA)", im0) 
 
        if torch.cuda.is_available() and (frame_id%30 == 0): 
            torch.cuda.empty_cache() 
            torch.cuda.ipc_collect() 
 
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break 
 
    cap.release() 
    cv2.destroyAllWindows() 
 
 
if __name__ == "__main__": 
    weights = "data/weights/yolopv2.pt" 
    detect_yolopv2_camera(weights)