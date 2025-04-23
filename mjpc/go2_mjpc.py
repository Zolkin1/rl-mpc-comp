import pathlib
from mujoco_mpc import agent as agent_lib

from mjpc_wrapper import MJPCWrapper

class Go2MJPC(MJPCWrapper) :
    def __init__(self):
        mujoco_mpc_path = pathlib.Path(agent_lib.__file__).resolve().parent
        model_path = (
            mujoco_mpc_path / "mjpc/tasks/quadruped/task_flat.xml"
        )
        super().__init__("Quadruped Flat", str(model_path))

