from setuptools import setup

package_name = "vls_ros2_bridge"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="user",
    maintainer_email="user@example.com",
    description="RL policy bridge between /h_marker/position and MAVROS velocity setpoints",
    license="MIT",
    entry_points={
        "console_scripts": [
            "landing_rl_node = vls_ros2_bridge.landing_rl_node:main",
        ],
    },
)
