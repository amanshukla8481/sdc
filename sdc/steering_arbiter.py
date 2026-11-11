#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32, String
from std_msgs.msg import Float32MultiArray
import time


class SteeringArbiter(Node):
    def __init__(self):
        super().__init__('steering_arbiter')

        # Parameters
        self.declare_parameter('override_duration', 3.0)
        self.declare_parameter('stop_distance_m', 0.5)  # if any lidar sector < this, treat as obstacle

        self.override_duration = float(self.get_parameter('override_duration').value)
        self.stop_distance = float(self.get_parameter('stop_distance_m').value)

        # State
        self.cam_steer = 0
        self.gps_steer = 0
        self.final_steer = 0
        self.override_until = 0.0
        self.obstacle_detected = False

        # Publishers/subscribers
        self.pub = self.create_publisher(Int32, '/final_steering', 10)
        self.create_subscription(Int32, '/steering_angle', self.cam_cb, 10)
        self.create_subscription(Int32, '/gps_steering', self.gps_cb, 10)
        self.create_subscription(String, '/navigation_decision', self.nav_cb, 10)
        self.create_subscription(Float32MultiArray, '/lidar_sectors', self.lidar_cb, 10)

        self.timer = self.create_timer(0.05, self.tick)
        self.get_logger().info('Steering arbiter started')

    def cam_cb(self, msg: Int32):
        try:
            self.cam_steer = int(msg.data)
        except Exception:
            pass

    def gps_cb(self, msg: Int32):
        try:
            self.gps_steer = int(msg.data)
        except Exception:
            pass

    def nav_cb(self, msg: String):
        # navigation decision like 'left' 'right' 'straight'
        now = time.time()
        if msg.data in ('left', 'right'):
            # override steering with gps for a fixed duration
            self.override_until = now + self.override_duration
            self.get_logger().info(f'Override requested: {msg.data} for {self.override_duration}s')

    def lidar_cb(self, msg: Float32MultiArray):
        # simple obstacle detection: if any sector within stop_distance -> obstacle
        try:
            sectors = list(msg.data)
            self.obstacle_detected = any(d < self.stop_distance for d in sectors if d > 0)
        except Exception:
            self.obstacle_detected = False

    def tick(self):
        now = time.time()
        if self.obstacle_detected:
            # override everything and stop steering (or zero steer)
            out = 0
        elif now < self.override_until:
            out = int(self.gps_steer)
        else:
            out = int(self.cam_steer)

        # publish final steering
        m = Int32()
        m.data = out
        self.pub.publish(m)


def main(args=None):
    rclpy.init(args=args)
    node = SteeringArbiter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
