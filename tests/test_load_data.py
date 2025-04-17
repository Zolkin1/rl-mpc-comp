import numpy as np

from motion_data.load_data import DataLoader

def test_load_data():
    dl = DataLoader("/home/zolkin/AmberLab/datasets/LAFAN1_Retargeting_Dataset/g1/walk1_subject1.csv")

    dl.create_segment("test_segment", 0, 100)

    diff = dl.get_config_idx(0) - [0.000480,-0.000023,0.796553,0.001059,0.016020,0.018009,0.999709,-0.119781,0.045812,
                             0.051567,0.283111,-0.137652,-0.030908,-0.078973,-0.066817,-0.237199,0.264636,-0.167496,
                             0.033785,0.021512,0.003775,-0.017551,-0.028931,1.667765,-0.055076,1.405656,0.068494,
                             0.181915,0.018075,0.048623,-1.620647,-0.084940,1.499361,0.098212,0.173991,-0.290950]

    assert np.all(diff == 0)

    diff = dl.get_config_idx(0) - dl.get_config_seg_idx("test_segment", 0)
    assert np.all(diff == 0)

