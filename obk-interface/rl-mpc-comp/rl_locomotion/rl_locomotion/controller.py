import os
import torch
import numpy as np
from abc import ABC, abstractmethod
from ament_index_python.packages import get_package_share_directory

from obelisk_control_msgs.msg import PDFeedForward
from obelisk_estimator_msgs.msg import EstimatedState
from obelisk_control_msgs.msg import VelocityCommand
from rclpy.lifecycle import LifecycleState, TransitionCallbackReturn

from obelisk_py.core.control import ObeliskController
from obelisk_py.core.obelisk_typing import ObeliskControlMsg, is_in_bound


class VelocityTrackingController(ObeliskController, ABC):
    """Example position setpoint controller."""

    def __init__(self, node_name: str = "velocity_tracking_controller") -> None:
        """Initialize the example position setpoint controller."""
        super().__init__(node_name, PDFeedForward, EstimatedState)
        # Velocity limits
        self.declare_parameter("v_x_max", 0.5)
        self.declare_parameter("v_x_min", -0.5)
        self.declare_parameter("v_y_max", 0.5)
        self.declare_parameter("w_z_max", 0.5)

        # Load policy
        self.declare_parameter("policy_name", "")
        policy_name = self.get_parameter("policy_name").get_parameter_value().string_value
        pkg_path = get_package_share_directory('rl_locomotion')
        policy_path = os.path.join(pkg_path, f'resource/policies/{policy_name}')
        self.policy = torch.load(policy_path)
        self.device = next(self.policy.parameters()).device

        # Set action scale, number of robot joints
        self.declare_parameter("action_scale", 0.25)
        self.action_scale = self.get_parameter("action_scale").get_parameter_value().double_value
        self.declare_parameter("num_motors", 12)
        self.num_motors = self.get_parameter("num_motors").get_parameter_value().integer_value

        # Set PD gains
        self.declare_parameter("kps", [25.] * self.num_motors)
        self.declare_parameter("kds", [0.5] * self.num_motors)
        self.kps = self.get_parameter("kps").get_parameter_value().double_array_value
        self.kds = self.get_parameter("kds").get_parameter_value().double_array_value

        # Phase info
        self.declare_parameter("phase_period", 0.3)
        self.phase_period = self.get_parameter("phase_period").get_parameter_value().double_value
        self.declare_parameter("stand_threshold", 0.01)
        self.stand_threshold = self.get_parameter("stand_threshold").get_parameter_value().double_value
        self.declare_parameter("no_phase_during_stand", True)
        self.no_phase_during_stand = self.get_parameter("no_phase_during_stand").get_parameter_value().bool_value
        self.standing = True
        self.last_time_to_stand = 0

        # Get default angles
        self.joint_names_isaac = []
        self.joint_names_mujoco = []
        self.declare_parameter("default_angles_isaac", [])  # Default angles in IsaacSim order
        self.default_angles_isaac = np.array(self.get_parameter("default_angles_isaac").get_parameter_value().double_array_value)
        
        # Declare subscriber to velocity commands
        self.register_obk_subscription(
            "sub_vel_cmd_setting",
            self.vel_cmd_callback,  # type: ignore
            key="sub_vel_cmd_key",  # key can be specified here or in the config file
            msg_type=VelocityCommand
        )

        self.get_logger().info(f"Policy: {policy_name} loaded on {self.device}. {len(self.kps)}, {len(self.kds)}")
        self.get_logger().info("RL Velocity Tracking node constructor complete.")

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Configure the controller."""
        super().on_configure(state)

        assert len(self.joint_names_isaac) == self.num_motors, f"Number of motors {self.num_motors} does not match number of isaac joint names {len(self.joint_names_isaac)}."
        assert len(self.joint_names_mujoco) == self.num_motors, f"Number of motors {self.num_motors} does not match number of mujoco joint names {len(self.joint_names_mujoco)}."
        assert self.default_angles_isaac.shape == (self.num_motors,), f"Default angles {self.default_angles_isaac} does not match number of motors {self.num_motors}."

        self.joint_pos = self.default_angles_mujoco.copy()
        self.joint_vel = np.zeros((self.num_motors,))
        self.cmd_vel = np.zeros((3,))
        self.proj_g = np.zeros((3,))
        self.proj_g[2] = -1
        self.omega = np.zeros((3,))
        self.phase = np.zeros((2,))
        self.zero_action = np.zeros((self.num_motors,))
        self.action = self.zero_action.tolist()
        self.t_start = None

        self.mujoco_to_isaac = [self.joint_names_mujoco.index(joint_name) for joint_name in self.joint_names_isaac]
        self.isaac_to_mujoco = [self.joint_names_isaac.index(joint_name) for joint_name in self.joint_names_mujoco]
        self.default_angles_mujoco = self.default_angles_isaac[self.isaac_to_mujoco]
        
        return TransitionCallbackReturn.SUCCESS

    def update_x_hat(self, x_hat_msg: EstimatedState) -> None:
        """Update the state estimate.

        Parameters:
            x_hat_msg: The Obelisk message containing the state estimate.
        """
        if len(x_hat_msg.q_joints) == self.num_motors:
            self.joint_pos = np.array(x_hat_msg.q_joints)
        else:
            self.get_logger().error(f"Estimated State joint position size does not match URDF! Size is {len(x_hat_msg.q_joints)} instead of {self.num_motors}.")

        if len(x_hat_msg.v_joints) == self.num_motors:
            self.joint_vel = np.array(x_hat_msg.v_joints)
        else:
            self.get_logger().error(f"Estimated State joint velocity size does not match URDF! Size is {len(x_hat_msg.v_joints)} instead of {self.num_motors}.")

        if len(x_hat_msg.q_base) == 7:
            self.proj_g = self.project_gravity(x_hat_msg.q_base[3:7])
        else:
            self.get_logger().error(f"Estimated State base pose size does not match URDF! Size is {len(x_hat_msg.q_base)} instead of 7.")

        if len(x_hat_msg.v_base) == 6:
            self.omega = np.array(x_hat_msg.v_base[3:6])
        else:
            self.get_logger().error(f"Estimated State base velocity size does not match URDF! Size is {len(x_hat_msg.v_base)} instead of 6.")

        
        t = x_hat_msg.header.stamp.sec + x_hat_msg.header.stamp.nanosec * 1e-9
        
        if self.no_phase_during_stand and np.linalg.norm(self.cmd_vel) <= self.stand_threshold:
            # If transitioning to stand, stop phase variable at next integer period
            if not self.standing:
                self.standing = True
                self.last_time_to_stand = np.ceil(t / self.phase_period) * self.phase_period
            theta = 2 * np.pi / self.phase_period * min(t - self.last_time_to_stand, 0)
        else:
            # If transitioning from stand, start phase variable at an integer period
            if np.linalg.norm(self.cmd_vel) > self.stand_threshold and self.standing:
                self.standing = False
                self.last_time_to_stand = t
            theta = 2 * np.pi / self.phase_period * (t - self.last_time_to_stand)
        self.phase = np.array([np.cos(theta), np.sin(theta)])

    def vel_cmd_callback(self, cmd_msg: VelocityCommand):
        self.cmd_vel[0] = min(
            max(cmd_msg.v_x, self.get_parameter("v_x_min").get_parameter_value().double_value),
            self.get_parameter("v_x_max").get_parameter_value().double_value
        )
        v_y_max = self.get_parameter("v_y_max").get_parameter_value().double_value
        self.cmd_vel[1] = min(max(cmd_msg.v_y, -v_y_max), v_y_max)
        w_z_max = self.get_parameter("w_z_max").get_parameter_value().double_value
        self.cmd_vel[2] = min(max(cmd_msg.w_z, -w_z_max), w_z_max)

    @staticmethod
    def project_gravity(quat):
        qx = quat[0]
        qy = quat[1]
        qz = quat[2]
        qw = quat[3]

        pg = np.zeros(3)

        pg[0] = 2 * (-qz * qx + qw * qy)
        pg[1] = -2 * (qz * qy + qw * qx)
        pg[2] = 1 - 2 * (qw * qw + qz * qz)

        return pg

    @abstractmethod
    def get_obs(self) -> np.ndarray:
        pass
    
    def compute_control(self) -> PDFeedForward:
        """Compute the control signal for the dummy 2-link robot.

        Returns:
            obelisk_control_msg: The control message.
        """
        # Generate input to RL model
        obs = self.get_obs()

        # Call RL model
        self.action = self.policy(torch.tensor(obs).to(self.device).float()).detach().cpu().numpy()

        # setting the message
        pd_ff_msg = PDFeedForward()
        pd_ff_msg.header.stamp = self.get_clock().now().to_msg()
        pos_targ = self.action[self.isaac_to_mujoco] * self.action_scale + self.default_angles_mujoco
        pd_ff_msg.pos_target = pos_targ.tolist()
        pd_ff_msg.vel_target = self.zero_action.tolist()
        pd_ff_msg.feed_forward = self.zero_action.tolist()
        pd_ff_msg.u_mujoco = np.concatenate([
            pos_targ,
            self.zero_action,
            self.zero_action
        ]).tolist()
        pd_ff_msg.joint_names = self.joint_names_mujoco
        pd_ff_msg.kp = self.kps
        pd_ff_msg.kd = self.kds
        self.obk_publishers["pub_ctrl"].publish(pd_ff_msg)
        assert is_in_bound(type(pd_ff_msg), ObeliskControlMsg)
        return pd_ff_msg
    
    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        return TransitionCallbackReturn.SUCCESS
