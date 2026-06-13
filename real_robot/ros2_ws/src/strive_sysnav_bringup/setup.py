from glob import glob
import os

from setuptools import find_packages, setup


package_name = "strive_sysnav_bringup"

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
    description="Launch files for STRIVE vendored SysNav detector and semantic mapping stack.",
    license="BSD-3-Clause",
)
