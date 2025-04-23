import threading
import pathlib

import numpy as np
import mujoco
from mujoco_mpc import agent as agent_lib


class MJPCWrapper:
    def __init__(self, task_id: str, model_path: str):
        self.task_id = task_id
        self.model_path = model_path

        self.model =  mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)

        self.u_lock = threading.Lock()
        self.x_lock = threading.Lock()

        self.u = self.data.ctrl
        self.qpos = self.data.qpos
        self.qvel = self.data.qvel

        self.mjpc_thread = threading.Thread(target=self.launch_mjpc)
        self.mjpc_thread.start()

    def update_state(self, qpos: np.array, qvel: np.array) -> None:
        # self.x_lock.acquire()
        self.qpos = qpos
        self.qvel = qvel
        # self.x_lock.release()

    def get_action(self) -> None:
        self.u_lock.acquire()
        u = self.u
        self.u_lock.release()

        return u

    def launch_mjpc(self) -> None:
        """Launch the MJPC server."""
        with agent_lib.Agent(
            server_binary_path=pathlib.Path(agent_lib.__file__).parent
                               / "mjpc"
                               / "ui_agent_server",
            task_id=self.task_id,
            model=self.model,
        ) as agent:
            while True:
                # self.update_state()
                agent.set_state(qpos=self.qpos, qvel=self.qvel)

                self.u_lock.acquire()
                self.u = agent.get_action()
                # print(self.u)
                self.u_lock.release()