
Build:
```
colcon build --symlink-install --parallel-workers $(nproc)
```

Source ROS2
```
source install/setup.bash
```
To launch the go2 sim:
```
obk-launch config_file_path=$RL_MPC_COMP_ROOT/rl-mpc-comp/mpc_locomotion/configs/go2_sim_mpc.yaml device_name=onboard bag=false
```

To run the go2 rl controller:
```
obk-launch config_file_path=$RL_MPC_COMP_ROOT/rl-mpc-comp/rl_locomotion/configs/go2_rl_vel_tracking.yml device_name=onboard bag=false
```