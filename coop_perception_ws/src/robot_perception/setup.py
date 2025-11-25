from setuptools import find_packages, setup

package_name = 'robot_perception'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='cmpe',
    maintainer_email='cmpe@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'dummy_detector = robot_perception.dummy_detector:main',
            'yolo2d_detector = robot_perception.yolo2d_detector:main',
            'lift_2d_to_3d = robot_perception.yolo_lift_3d:main',
            'lift_2d_to_3d_sync = robot_perception.yolo_lift_3d_sync:main',
            'yolo_lift_3d_sync_backup = robot_perception.yolo_lift_3d_sync_backup:main',
            'lift_2d_to_3d_latest = robot_perception.yolo_lift_3d_latest:main',
            'to_map_republisher = robot_perception.to_map_republisher:main',
            'fuse_concat = robot_perception.fuse_concat:main',
            'fuse_distance = robot_perception.fuse_distance:main',
            'fuse_merge_robust = robot_perception.fuse_merge_robust:main',
            'awareness_bridge = robot_perception.awareness_bridge:main',
            'metrics_logger = robot_perception.metrics_logger:main',
            'accuracy_proxy = robot_perception.accuracy_proxy:main',
            'accuracy_proxy_match = robot_perception.accuracy_proxy_match:main',
        ],
    },
)
