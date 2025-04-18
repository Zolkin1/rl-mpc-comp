import argparse

import mujoco

from hydrax.algs import CEM
from hydrax.simulation.deterministic import run_interactive

from motion_data.load_data import DataLoader
from sampling_mpc.walk_mirror_task import SegmentMirrorTask

import jax.numpy as jnp
import numpy as np

"""
Run an interactive simulation of the humanoid motion capture tracking task.
"""

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Run an interactive simulation of mocap tracking with the G1."
)
parser.add_argument(
    "--reference_filename",
    type=str,
    default="walk1_subject1.csv",
)
parser.add_argument(
    "--dataset_dir",
    type=str,
    help="Location of the dataset",
)
args = parser.parse_args()

# Define the task (cost and dynamics)
file_location = args.dataset_dir + "/" + args.reference_filename
dl = DataLoader(file_location)
seg_start = 327
seg_end = 559
dl.create_segment("walking_seg", seg_start, seg_end)
task = SegmentMirrorTask(dl, "walking_seg")

# Set up the controller
ctrl = CEM(
    task,
    num_samples=512, # 1024
    num_elites=20,
    sigma_start=0.1,
    sigma_min=0.1,
    plan_horizon=0.6,
    spline_type="zero",
    num_knots=4,
)

# Define the model used for simulation
mj_model = task.mj_model
mj_model.opt.timestep = 0.001 #0.01
mj_model.opt.iterations = 10
mj_model.opt.ls_iterations = 50
mj_model.opt.o_solimp = [0.9, 0.95, 0.001, 0.5, 2]
mj_model.opt.enableflags = mujoco.mjtEnableBit.mjENBL_OVERRIDE

# Convert the dataset to mujoco format, with wxyz quaternion
pos = dl.data[:, :3]
xyzw = dl.data[:, 3:7]
wxyz = np.concatenate([xyzw[:, 3:], xyzw[:, :3]], axis=1)
reference = np.concatenate([pos, wxyz, dl.data[:, 7:]], axis=1)
reference[seg_start:seg_end, :2] = reference[seg_start:seg_end, :2] - reference[seg_start, :2]

# Set the initial state
mj_data = mujoco.MjData(mj_model)
mj_data.qpos[:] = reference[seg_start, :]


run_interactive(
    ctrl,
    mj_model,
    mj_data,
    frequency=100,
    show_traces=False,
    reference=reference[seg_start:seg_end, :],  # TODO: Adjust the data here to have the right quaternion convention
)