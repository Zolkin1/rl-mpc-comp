from typing import List, Optional

from rclpy.executors import SingleThreadedExecutor
from obelisk_py.core.utils.ros import spin_obelisk
import numpy as np
from obelisk_control_msgs.msg import PDFeedForward
from obelisk_estimator_msgs.msg import EstimatedState

from rl_locomotion.controller import VelocityTrackingController


class Go2VelocityTrackingController(VelocityTrackingController):
    """Example position setpoint controller."""

    def __init__(self, node_name: str = "velocity_tracking_controller") -> None:
        """Initialize the example position setpoint controller."""
        super().__init__(node_name, PDFeedForward, EstimatedState)

        # Get default angles
        self.joint_names_isaac = [
            "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
            "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
            "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint"
        ]
        self.joint_names_mujoco = [
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"
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