from typing import List, Optional

import numpy as np
from obelisk_control_msgs.msg import PDFeedForward
from obelisk_estimator_msgs.msg import EstimatedState
from rclpy.lifecycle import LifecycleState, TransitionCallbackReturn

from obelisk_py.core.control import ObeliskController
from obelisk_py.core.obelisk_typing import ObeliskControlMsg, ObeliskEstimatorMsg, is_in_bound

from rclpy.executors import SingleThreadedExecutor

from obelisk_py.core.utils.ros import spin_obelisk
from obelisk_py.zoo.control.example.example_position_setpoint_controller import ExamplePositionSetpointController



class ExamplePositionSetpointController(ObeliskController):
    """Example position setpoint controller."""

    def __init__(self, node_name: str = "example_position_setpoint_controller") -> None:
        """Initialize the example position setpoint controller."""
        super().__init__(node_name, PDFeedForward, EstimatedState)

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Configure the controller."""
        super().on_configure(state)
        self.x_hat = None
        return TransitionCallbackReturn.SUCCESS

    def update_x_hat(self, x_hat_msg: ObeliskEstimatorMsg) -> None:
        """Update the state estimate.

        Parameters:
            x_hat_msg: The Obelisk message containing the state estimate.
        """
        pass  # do nothing

    def compute_control(self) -> ObeliskControlMsg:
        """Compute the control signal for the dummy 2-link robot.

        Returns:
            obelisk_control_msg: The control message.
        """
        # # computing the control input
        # u = np.sin(self.t)  # example state-independent control input

        # # setting the message
        # position_setpoint_msg = PDFeedForward()
        # position_setpoint_msg.u = [u]
        # self.obk_publishers["pub_ctrl"].publish(position_setpoint_msg)
        # assert is_in_bound(type(position_setpoint_msg), ObeliskControlMsg)
        # return position_setpoint_msg  # type: ignore

def main(args: Optional[List] = None) -> None:
    """Main entrypoint."""
    spin_obelisk(args, ExamplePositionSetpointController, SingleThreadedExecutor)


if __name__ == "__main__":
    main()