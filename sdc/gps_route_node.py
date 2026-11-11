#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_msgs.msg import Int32
from sensor_msgs.msg import NavSatFix
import math
import time


def haversine(lat1, lon1, lat2, lon2):
    # returns distance in meters between two lat/lon points
    R = 6371000.0  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi/2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2.0)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c


class GPSRouteNode(Node):

    def __init__(self):
        super().__init__('gps_route_node')

        # Parameters
        self.declare_parameter('junction_radius_m', 10.0)
        self.declare_parameter('steering_threshold_deg', 15)

        self.junction_radius = float(self.get_parameter('junction_radius_m').value)
        self.steering_threshold = float(self.get_parameter('steering_threshold_deg').value)

        # Publisher for high-level nav decision
        self.decision_pub = self.create_publisher(String, '/navigation_decision', 10)

        # Subscriptions
        self.gps_sub = self.create_subscription(NavSatFix, '/gps/fix', self.gps_callback, 10)
        self.steer_sub = self.create_subscription(Int32, '/steering_angle', self.steer_callback, 10)

        # State
        self.current_lat = None
        self.current_lon = None
        self.latest_steering = 0
        self.last_decision_time = 0.0
        self.decision_cooldown = 2.0  # seconds between decision publishes at same junction

        # Waypoints (simple example). Each waypoint can be marked as 'junction': True/False
        # In production, load from parameter or file. Coordinates are (lat, lon).
        # Replace these with your actual route waypoints.
        self.waypoints = [
            {'lat': 37.4219999, 'lon': -122.0840575, 'junction': False},
            {'lat': 37.4225, 'lon': -122.0845, 'junction': True},
            {'lat': 37.4230, 'lon': -122.0850, 'junction': False},
        ]

        self.get_logger().info('GPS Route Node started')

    def steer_callback(self, msg: Int32):
        try:
            self.latest_steering = int(msg.data)
        except Exception:
            self.get_logger().warn('Received bad steering message')

    def gps_callback(self, msg: NavSatFix):
        # Update current GPS
        if msg.status.status < 0:
            # invalid fix
            return

        self.current_lat = float(msg.latitude)
        self.current_lon = float(msg.longitude)

        # Check junctions
        self.check_nearby_junctions()

    def check_nearby_junctions(self):
        if self.current_lat is None or self.current_lon is None:
            return

        now = time.time()

        for wp in self.waypoints:
            if not wp.get('junction', False):
                continue

            dist = haversine(self.current_lat, self.current_lon, wp['lat'], wp['lon'])
            if dist <= self.junction_radius:
                # At a junction — decide branch using steering input
                # Prevent spamming the same decision repeatedly
                if now - self.last_decision_time < self.decision_cooldown:
                    return

                decision = self.decide_branch(self.latest_steering)
                msg = String()
                msg.data = decision
                self.decision_pub.publish(msg)
                self.get_logger().info(f"At junction (dist {dist:.1f} m), steering={self.latest_steering} => decision={decision}")
                self.last_decision_time = now
                return

    def decide_branch(self, steering_angle_deg: float) -> str:
        # Simple rule: strongly positive => right, strongly negative => left, else straight
        if steering_angle_deg > self.steering_threshold:
            return 'right'
        elif steering_angle_deg < -self.steering_threshold:
            return 'left'
        else:
            return 'straight'


def main(args=None):
    rclpy.init(args=args)
    node = GPSRouteNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
