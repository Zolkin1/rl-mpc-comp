
Build:
```
colcon build --symlink-install --parallel-workers $(nproc)
```

To launch the go2 sim:
```
obk-launch config_file_path=$RL_MPC_COMP_ROOT/rl-mpc-comp/mpc_locomotion/configs/go2_sim_mpc.yaml device_name=onboard bag=false
```