#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from cv_bridge import CvBridge
import cv2

import time
import torch
import numpy as np
import math
import os

from ament_index_python.packages import get_package_share_directory

from collect_lane_points import collect_lane_points
from find_desired_angle import find_desired_angle
from main_lanes_detect import extract_main_lanes
from poly_fit import poly_fit_lane

from utils.utils import (
    select_device, time_synchronized, lane_line_mask,
    show_seg_result, letterbox
)

import image_enhancement as ie


class YOLOPv2LaneNode(Node):

    def __init__(self):
        super().__init__("yolopv2_lane_detector")

        self.bridge = CvBridge()

        # ---------------------------------------------------------------
        # FIXED: Load model using ROS package share directory
        # ---------------------------------------------------------------
        package_share = get_package_share_directory("sdc")
        weight_path = os.path.join(
            package_share, "data", "weights", "yolopv2.pt"
        )

        device = select_device('')
        self.model = torch.jit.load(weight_path, map_location=device)
        # ---------------------------------------------------------------

        self.device = device
        self.model.to(device).eval()

        self.half = device.type != "cpu"
        if self.half:
            self.model.half()

        self.prev_center_line = None
        self.frame_id = 0

        # Subscribe to the camera
        self.create_subscription(
            Image,
            "/image_raw",
            self.image_callback,
            10
        )

        # Publish steering angle
        self.angle_pub = self.create_publisher(Int32, "/steering_angle", 10)

        self.get_logger().info("YOLOPv2 Lane Detector Node Started")

    def image_callback(self, msg: Image):
        # Convert ROS image → OpenCV
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        # Resize exactly like your original code
        frame = cv2.resize(frame, (1280, 720))

        # Enhance
        frame = ie.enhance_image_for_sunlight_glare(frame)

        im0 = frame.copy()
        h, w = 720, 1280

        # Mask top
        im0[:h // 2 - 120, :] = 0

        img = letterbox(im0, 640, stride=32)[0]

        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)

        img = img.half() if self.half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        with torch.no_grad():
            t1 = time_synchronized() 
            [pred, anchor_grid], seg, ll = self.model(img) 
            t2 = time_synchronized() 

        ll_seg_mask = lane_line_mask(ll)
        da_seg_mask = np.zeros_like(ll_seg_mask)

        show_seg_result(im0, (da_seg_mask, ll_seg_mask), is_demo=True)

        lane_points = extract_main_lanes(ll_seg_mask)

        y_h, x_w = ll_seg_mask.shape
        y_line = y_h // 2 + 50

        visited = set()
        lanes_data = []

        for x in lane_points:
            if x > 0 and ll_seg_mask[y_line, x] == 255 and (y_line, x) not in visited:
                lane_pix = collect_lane_points(ll_seg_mask, y_line, x)
                lanes_data.append(lane_pix)

        poly_fits = []
        for lane in lanes_data:
            lane = np.array(lane)
            if len(lane) < 20:
                continue
            coeffs = poly_fit_lane(lane, degree=2)
            poly_fits.append(coeffs)

        center_line = None

        if len(poly_fits) == 2:
            center_line = (poly_fits[0] + poly_fits[1]) / 2
            self.prev_center_line = center_line
            poly_fits.append(center_line)

        elif len(poly_fits) == 3:
            center_line = poly_fits[2]
            self.prev_center_line = center_line

        elif self.prev_center_line is not None:
            center_line = self.prev_center_line
            poly_fits.append(center_line)

        # Draw lane fits
        colors = [(0,255,0),(0,255,0),(255,0,0),(255,255,0)]
        for idx, coeffs in enumerate(poly_fits):
            plot_y = np.linspace(y_h//3, y_h-1, 100)
            x_vals = np.polyval(coeffs, plot_y)
            pts = np.array([np.column_stack((x_vals, plot_y))], dtype=np.int32)
            cv2.polylines(im0, pts, False, colors[idx % len(colors)], 3)

        # Compute steering angle
        if center_line is not None:

            signed_angle_deg, ang_AB, ang_AC = find_desired_angle(
                center_line, y_h, x_w
            )

            signed_angle_deg = np.clip(signed_angle_deg, -80, 80)

            # PUBLISH angle
            msg = Int32()
            msg.data = int(signed_angle_deg)
            self.angle_pub.publish(msg)

            # Visualization
            x_car = x_w//2
            y_car = y_h-1
            car_pos = (x_car, y_car)

            y_dest = y_h//2 + 80
            x_dest = int(np.polyval(center_line, y_dest))
            dest_pos = (x_dest, int(y_dest))
            c_pos = (x_car, int(y_dest))

            text = f"signed: {signed_angle_deg:+.1f} deg"
            cv2.putText(im0, text, (x_car - 150, y_car - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255,255,0),2)

            start_deg = math.degrees(ang_AC) % 360
            end_deg   = math.degrees(ang_AB) % 360

            cv2.ellipse(im0, car_pos, (40,40), 0,
                        start_deg, end_deg,
                        (0,255,255), 2)

            cv2.line(im0, car_pos, dest_pos, (255,0,255), 2)
            cv2.line(im0, car_pos, c_pos, (0,255,0),     2)

            cv2.circle(im0, car_pos, 5, (0,0,255), -1)
            cv2.circle(im0, dest_pos, 4, (255,0,255), -1)
            cv2.circle(im0, c_pos,   4, (0,255,0), -1)

        # Show FPS
        fps = 1 / (t2 - t1)
        cv2.putText(im0, f"FPS: {fps:.1f}", (20,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (0,255,255),2)

        cv2.imshow("YOLOPv2 Lane (ROS2)", im0)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = YOLOPv2LaneNode()
    rclpy.spin(node)
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
