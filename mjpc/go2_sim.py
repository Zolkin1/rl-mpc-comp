
from go2_mjpc import Go2MJPC
import sync_sim

if __name__ == '__main__':
    go2_ctrl = Go2MJPC()
    sync_sim.run_sim(ctrl=go2_ctrl, ctrl_freq=100.)