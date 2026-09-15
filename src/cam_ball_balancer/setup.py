import os
from glob import glob
from setuptools import find_packages, setup

package_name = "cam_ball_balancer"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Caleb Churata",
    maintainer_email="caleb.churata@utec.edu.pe",
    description="DSM (MT5001) ball-on-plate vision: ball tracking + ArUco platform pose.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "ball_tracker_node = cam_ball_balancer.ball_tracker_node:main",
            "platform_pose_node = cam_ball_balancer.platform_pose_node:main",
        ],
    },
)
