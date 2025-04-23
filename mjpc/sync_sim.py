import time
import mujoco
import mujoco.viewer

def run_sim(
        ctrl,
        ctrl_freq: float = 100.,
) -> None:
    """Simulate an MJPC controller in an asynchronous manner."""
    print("Running the sim...")

    mj_model = ctrl.model
    mj_data = ctrl.data

    sim_dt = mj_model.opt.timestep

    sim_steps_per_ctrl = int((1/ctrl_freq) / sim_dt)

    print("About to launch viewer...")
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        print("Viewer launched.")
        while viewer.is_running:
            start_time = time.time()

            ctrl.update_state(qpos=mj_data.qpos, qvel=mj_data.qvel)
            u = ctrl.get_action()

            for i in range(sim_steps_per_ctrl):
                mj_data.ctrl[:] = u
                mujoco.mj_step(mj_model, mj_data)
                # print("Stepping")
                viewer.sync()

            while time.time() - start_time < 1./ctrl_freq:
                time.sleep(1./ctrl_freq - (time.time() - start_time))

    print("Viewer closed.")
