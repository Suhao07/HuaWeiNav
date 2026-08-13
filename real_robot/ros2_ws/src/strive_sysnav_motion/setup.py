from setuptools import find_packages, setup
from glob import glob
import os


package_name = "strive_sysnav_motion"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="STRIVE Real Robot Integration",
    maintainer_email="devnull@example.com",
    description="Task-level SysNav waypoint motion server for STRIVE.",
    license="BSD-3-Clause",
    entry_points={
        "console_scripts": [
            "sysnav_motion_server = strive_sysnav_motion.motion_server:main",
            "safety_velocity_mux = strive_sysnav_motion.safety_velocity_mux:main",
            "motion_hil = strive_sysnav_motion.motion_hil:main",
            "lower_bag_probe = strive_sysnav_motion.lower_bag_probe:main",
        ],
    },
)
