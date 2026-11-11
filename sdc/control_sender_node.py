#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
from std_msgs.msg import Float32MultiArray
import serial
import time

class ControlSenderNode(Node):

    def __init__(self):
        super().__init__("control_sender_node")

        # --- Serial to Arduino ---
        try:
            self.arduino = serial.Serial('/dev/ttyACM0', 115200, timeout=0)  
            time.sleep(2)
            self.get_logger().info("Arduino connected on /dev/ttyACM0")
        except:
            self.arduino = None
            self.get_logger().error("Failed to connect to Arduino")

        # Stored data
        self.latest_angle = 0
        self.latest_final_angle = None
        self.obstacle_detected = 0

        # Subscribers
        # Subscribe to camera steering and final (arbitrated) steering
        self.create_subscription(Int32, "/steering_angle", self.angle_callback, 10)
        self.create_subscription(Int32, "/final_steering", self.final_angle_callback, 10)
        self.create_subscription(Float32MultiArray, "/lidar_sectors", self.lidar_callback, 10)

        # Timer for sending and receiving (10 Hz)
        self.create_timer(0.1, self.serial_tick)

    # ---------------- SUBSCRIBERS ----------------
    def angle_callback(self, msg: Int32):
        self.latest_angle = int(msg.data)

    def final_angle_callback(self, msg: Int32):
        try:
            self.latest_final_angle = int(msg.data)
        except Exception:
            self.latest_final_angle = None

    def lidar_callback(self, msg: Float32MultiArray):
        sectors = msg.data
        self.obstacle_detected = 1 if any(d < 0.5 for d in sectors) else 0

    # ---------------- SERIAL HANDLER -------------
    def serial_tick(self):
        if not self.arduino:
            return

        # -------- SEND --------
        # Prefer final (arbitrated) steering when available
        send_angle = self.latest_final_angle if self.latest_final_angle is not None else self.latest_angle
        packet = f"A:{send_angle};O:{self.obstacle_detected}\n"
        try:
            self.arduino.write(packet.encode())
        except Exception:
            self.get_logger().error("Failed to send packet to Arduino")

        # -------- RECEIVE (NON BLOCKING) --------
        try:
            if self.arduino.in_waiting > 0:
                line = self.arduino.readline().decode().strip()
                if line != "":
                    self.get_logger().info(f"Arduino: {line}")
        except Exception as e:
            self.get_logger().error(f"Serial read error: {e}")

# ---------------- MAIN ----------------
def main(args=None):
    rclpy.init(args=args)
    node = ControlSenderNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
