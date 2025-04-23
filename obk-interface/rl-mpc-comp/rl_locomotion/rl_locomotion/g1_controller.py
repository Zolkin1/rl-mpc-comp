import numpy as np
from typing import List, Optional

from rclpy.executors import SingleThreadedExecutor
from obelisk_py.core.utils.ros import spin_obelisk

from rl_locomotion.controller import VelocityTrackingController


class Go2VelocityTrackingController(VelocityTrackingController):
    """Example position setpoint controller."""

    def __init__(self, node_name: str = "g1_velocity_tracking_controller") -> None:
        """Initialize the example position setpoint controller."""
        super().__init__(node_name)

        # Get default angles
        self.joint_names_isaac = [ # TODO: Joint names
        ]
        self.joint_names_mujoco = [ # TODO: Joint names
        ]

    def get_obs(self) -> np.ndarray:
        return np.concatenate([
            self.omega,
            self.proj_g,
            self.cmd_vel,
            (self.joint_pos - self.default_angles_mujoco)[self.mujoco_to_isaac],
            self.joint_vel[self.mujoco_to_isaac],
            self.action,
            self.phase
        ])
    

def main(args: Optional[List] = None) -> None:
    """Main entrypoint."""
    spin_obelisk(args, Go2VelocityTrackingController, SingleThreadedExecutor)


if __name__ == "__main__":
    main()