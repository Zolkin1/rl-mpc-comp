import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'rl_locomotion'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join("share", package_name, "launch"), glob(os.path.join("launch", "*launch.[pxy][yma]*"))),
        # this is how you expose data directories in your package
        (os.path.join("share", package_name, "config"), glob(os.path.join("config", "*.yaml"))),
        (os.path.join("share", package_name, "resource/policies"), glob(os.path.join("resource/policies", "*.pt"))),  # Get the models..
        (os.path.join("share", package_name, "config"), glob(os.path.join("config", "*.yml"))),
        (os.path.join("share", package_name, "config"), glob(os.path.join("config", "*.obk"))),
        (os.path.join("share", package_name, "rviz"), glob(os.path.join("rviz", "*.rviz"))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='wcompton',
    maintainer_email='wdc3iii@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "go2_controller = rl_locomotion.go2_controller:main",
            "g1_controller = rl_locomotion.g1_controller:main"
        ],
    },
)
