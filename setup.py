from setuptools import setup
import os
from glob import glob

package_name = 'sdc'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        # ----------------------------------------------------------
        # ADD THIS SECTION (installs your weights into share/sdc/)
        # ----------------------------------------------------------
        ('share/' + package_name + '/data/weights', [
            'data/weights/yolopv2.pt'
        ]),
        # ----------------------------------------------------------
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='robotics',
    maintainer_email='robotics@todo.todo',
    description='SDC Lidar detection package',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'lidar_detection = sdc.lidar_detection:main',
            'yolopv2_lane_detector = sdc.yolopv2_lane_detector:main',
            'control_sender_node = sdc.control_sender_node:main',
            'gps_route_node = sdc.gps_route_node:main',
            'mavlink_gps_bridge = sdc.mavlink_gps_bridge:main',
            'steering_arbiter = sdc.steering_arbiter:main'
        ],
    },
)

