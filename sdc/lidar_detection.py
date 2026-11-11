#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray
import math


class LidarSectorDetector(Node):

    def __init__(self):
        super().__init__("lidar_sector_detector")

        self.declare_parameter("distance_threshold", 10.0)
        self.distance_threshold = self.get_parameter("distance_threshold").value

        # Sector width = 15°
        self.sector_angle_width = math.radians(15)

        # Two back-side ranges:
        # A: [-π, -π/2]
        # B: [π/2, π]
        self.regions = [
            (-math.pi, -math.pi/2),
            ( math.pi/2,  math.pi)
        ]

        self.total_sectors = 12

        self.scan_sub = self.create_subscription(
            LaserScan, "/scan", self.scan_callback, 10
        )

        # ✔ FIXED: publish floats not strings
        self.pub = self.create_publisher(Float32MultiArray, "/lidar_sectors", 10)

        self.get_logger().info("Lidar 12-Sector Detector Started")

    def scan_callback(self, msg: LaserScan):

        ranges = msg.ranges
        angle_min = msg.angle_min
        angle_inc = msg.angle_increment

        # 12 empty sectors
        sectors = [[] for _ in range(self.total_sectors)]

        # Assign points to sectors
        for idx, r in enumerate(ranges):

            angle = angle_min + idx * angle_inc  # radians

            # Region A [-π, -π/2]
            if -math.pi <= angle < -math.pi/2:
                sector_idx = int((angle + math.pi) / self.sector_angle_width)

            # Region B [π/2, π]
            elif math.pi/2 <= angle <= math.pi:
                sector_idx = 6 + int((angle - math.pi/2) / self.sector_angle_width)

            else:
                continue

            if 0 <= sector_idx < 12:
                sectors[sector_idx].append(r)

        # ------------------------------
        # Compute min distance per sector
        # ------------------------------
        min_distances = []

        for i in range(12):
            valid = [d for d in sectors[i] if msg.range_min < d < msg.range_max]

            if not valid:
                min_distances.append(float("inf"))
            else:
                min_distances.append(min(valid))

        # ✔ Publish Float32MultiArray
        ros_msg = Float32MultiArray(data=min_distances)
        self.pub.publish(ros_msg)

        # Optional debug print
        print("\nLIDAR Sector Distances:", min_distances)


def main(args=None):
    rclpy.init(args=args)
    node = LidarSectorDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
