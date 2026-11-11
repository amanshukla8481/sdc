#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix, NavSatStatus
from pymavlink import mavutil
import threading
import time


class MAVLinkGPSBridge(Node):
    def __init__(self):
        super().__init__('mavlink_gps_bridge')

        # parameters
        self.declare_parameter('device', '/dev/ttyACM0')
        self.declare_parameter('baud', 115200)
        self.device = self.get_parameter('device').value
        self.baud = int(self.get_parameter('baud').value)

        self.pub = self.create_publisher(NavSatFix, '/gps/fix', 10)

        self.get_logger().info(f'Opening MAVLink on {self.device} @ {self.baud}')

        # Start mavlink receiver thread
        try:
            self.master = mavutil.mavlink_connection(self.device, baud=self.baud)
        except Exception as e:
            self.get_logger().error(f'Failed to open MAVLink device: {e}')
            raise

        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while rclpy.ok() and self._running:
            try:
                msg = self.master.recv_match(blocking=True, timeout=1)
                if msg is None:
                    continue
                mtype = msg.get_type()
                # GLOBAL_POSITION_INT: lat/lon in 1e7, relative_alt in mm
                if mtype == 'GLOBAL_POSITION_INT':
                    lat = msg.lat / 1e7
                    lon = msg.lon / 1e7
                    # relative_alt is millimeters (often)
                    try:
                        alt = msg.relative_alt / 1000.0
                    except Exception:
                        alt = 0.0
                    nav = NavSatFix()
                    nav.header.stamp = self.get_clock().now().to_msg()
                    nav.header.frame_id = 'gps'
                    nav.status.status = NavSatStatus.STATUS_FIX
                    nav.latitude = float(lat)
                    nav.longitude = float(lon)
                    nav.altitude = float(alt)
                    self.pub.publish(nav)

                elif mtype == 'GPS_RAW_INT':
                    # GPS_RAW_INT: lat/lon in 1e7, alt in mm
                    lat = msg.lat / 1e7
                    lon = msg.lon / 1e7
                    try:
                        alt = msg.alt / 1000.0
                    except Exception:
                        alt = 0.0
                    nav = NavSatFix()
                    nav.header.stamp = self.get_clock().now().to_msg()
                    nav.header.frame_id = 'gps'
                    nav.status.status = NavSatStatus.STATUS_FIX
                    nav.latitude = float(lat)
                    nav.longitude = float(lon)
                    nav.altitude = float(alt)
                    self.pub.publish(nav)

                # ignore other messages
            except Exception as e:
                self.get_logger().warn(f'MAVLink read error: {e}')
                time.sleep(0.5)

    def destroy_node(self):
        self._running = False
        try:
            if hasattr(self, 'master') and self.master is not None:
                self.master.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MAVLinkGPSBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
