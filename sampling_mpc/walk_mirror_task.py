import numpy as np
import mujoco
from hydrax import ROOT
from mujoco import mjx
import jax
import jax.numpy as jnp
from hydrax.task_base import Task

from motion_data.load_data import DataLoader

class SegmentMirrorTask(Task):
    def __init__(self, dl: DataLoader, segment_name: str):
        mj_model = mujoco.MjModel.from_xml_path(ROOT + "/models/g1/scene.xml")
        super().__init__(
            mj_model,
            trace_sites=["imu_in_torso", "left_foot", "right_foot"],
        )

        self.dl = dl
        self.segment_name = segment_name

        # Compute x-y velocities
        self.vel = jnp.diff(dl.data[:, :2], axis=0)
        self.vel = self.vel / 0.033333333

        # Cost weights
        cost_weights = np.ones(mj_model.nq - 2)     # No tracking on x-y position
        cost_weights[:5] = 10.0  # Base pose is more important
        self.cost_weights = jnp.array(cost_weights)
        self.vel_weight = 1*jnp.ones(2)        # x-y velocity tracking

    def running_cost(self, state: mjx.Data, control: jax.Array) -> float:
        # Implement the running cost (l) here
        # Configuration error weighs the base pose more heavily
        q_ref = self._get_reference_configuration(state.time)
        q = state.qpos
        q_err = self.cost_weights * (q[2:] - q_ref[2:])     # Ignore the x-y position
        configuration_cost = jnp.sum(jnp.square(q_err))

        # Base velocity cost
        v_ref = self._get_reference_vel(state.time)
        v = state.qvel
        # jax.debug.print("vref: {}, v: {}", v_ref, v)
        v_err = self.vel_weight * (v[:2] - v_ref[:2])
        vel_cost = jnp.sum(jnp.square(v_err))

        # Control penalty incentivizes driving toward the reference, since all
        # joints are position-controlled
        u_ref = q_ref[7:]
        control_cost = jnp.sum(jnp.square(control - u_ref))

        return 1.0 * configuration_cost + 1.0 * control_cost + 1.0 * vel_cost

    def terminal_cost(self, state: jax.Array) -> float:
        # Implement the terminal cost (phi) here
        q_ref = self._get_reference_configuration(state.time)
        q = state.qpos
        q_err = self.cost_weights * (q[2:] - q_ref[2:])
        # N.B. we multiply by dt to ensure the terminal cost is comparable to
        # the running cost, since this isn't a proper cost-to-go.
        return self.dt * jnp.sum(jnp.square(q_err))

    def _get_reference_configuration(self, time):
        """Get the reference."""
        # Using the current time, compute the time within the segment
        q, _ = self.dl.get_config_seg(self.segment_name, time)

        # Modify q to have the correct format
        pos = q[:3]
        xyzw = q[3:7]
        print(xyzw[3:])
        wxyz = jnp.concatenate([xyzw[3:], xyzw[:3]])
        return jnp.concatenate([pos, wxyz, q[7:]])

    def _get_reference_vel(self, time):
        """Get the reference velocity."""
        q, idx = self.dl.get_config_seg(self.segment_name, time)

        return self.vel[idx]