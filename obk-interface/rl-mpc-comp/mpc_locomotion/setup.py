import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'mpc_locomotion'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join("share", package_name, "config"), glob(os.path.join("config", "*.yaml"))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='zolkin',
    maintainer_email='zach.olkin@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "controller = mpc_locomotion.controller:main",
            "state_estimator = mpc_locomotion.state_estimator:main",
            "robot_sim = mpc_locomotion.robot_sim:main",
        ],
    },
)
