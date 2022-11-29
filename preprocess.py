# preprocess.py
# process LIDAR sweeps to identify ground returns
# this is required to identify freespace
import os
import argparse
import numpy as np
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from util.once_devkit.once import ONCE
from lib.grndseg import segmentation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
#
def process(sd):

    # read lidar points
    if dataset == 'nusc':
        pc = LidarPointCloud.from_file(f"{data_root}/{sd['filename']}")
        pts = np.array(pc.points[:3].T)
    elif dataset == 'once':
        pc = data.load_point_cloud(sd['seq_id'], sd['frame_id'])
        pts = np.array(pc[:, :3])
        pc = pc.T

    # we follow nuscenes's labeling protocol
    # 24: driveable surface
    # 30: static.others
    # 31: ego

    # initialize everything to 30 (static others)
    lbls = np.full(len(pts), 30, dtype=np.uint8)

    # identify ego mask based on the car's physical dimensions
    if dataset == 'nusc':
        ego_mask = np.logical_and(
            np.logical_and(-0.8 <= pc.points[0], pc.points[0] <= 0.8),
            np.logical_and(-1.5 <= pc.points[1], pc.points[1] <= 2.5)
        )
    elif dataset == 'once':
        ego_mask = np.logical_and(
            np.logical_and(-0.8 <= pc[0], pc[0] <= 0.8),
            np.logical_and(-2.5 <= pc[1], pc[1] <= 2.5)
        )

    lbls[ego_mask] = 31

    # run ground segmentation code
    index = np.flatnonzero(np.logical_not(ego_mask))
    if dataset == "nusc":
        label = segmentation.segment(pts[index])
    elif dataset == "once":
        label = segmentation.segment(pts[index],
                sensor_height=1.84, # no effect
                max_fit_error=0.1, # v small effect on changing to 0.15
                line_search_angle=1.5, # best results at 1.5
                )

    #
    grnd_index = np.flatnonzero(label)
    lbls[index[grnd_index]] = 24

    #
    if dataset == 'nusc':
        res_file = os.path.join(res_dir, f"{sd['token']}_grndseg.bin")
    elif dataset == 'once':
        res_dir_ = os.path.join(res_dir, f"{sd['seq_id']}/grndseg")
        if not os.path.exists(res_dir_):
            os.makedirs(res_dir_, exist_ok=True)
        res_file = os.path.join(res_dir_, f"{sd['frame_id']}.bin")
    lbls.tofile(res_file)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="/data/once")
    parser.add_argument("--data-version", type=str, default="train")
    parser.add_argument("--dataset", type=str, default="once", choices=['nusc', 'once'])
    parser.add_argument("--num-workers", type=int, default=16)
    args = parser.parse_args()

    # nusc_version = "v1.0-mini"
    data_version = args.data_version
    data_root = args.data_root
    dataset = args.dataset

    if dataset == 'nusc':
        data = NuScenes(data_version, data_root)
    elif dataset == 'once':
        data = ONCE(data_root)
    else:
        print('Not implemented!')
        exit(0)

    if dataset == 'nusc':
        res_dir = f"{data_root}/grndseg/{data_version}"
        if not os.path.exists(res_dir):
            os.makedirs(res_dir, exist_ok=True)
    elif dataset == 'once':
        res_dir = f"{data_root}/data"

    #
    if dataset == 'nusc':
        sds = [sd for sd in data.sample_data if sd["channel"] == "LIDAR_TOP"]
    elif dataset == 'once':
        sds = []
        split_info = getattr(data, f'{data_version}_info')
        scene_names = split_info.keys()
        for scene_name in scene_names:
            scene = split_info[scene_name]
            for frame in scene['frame_list']:
                scene[frame]['frame_id'] = frame
                scene[frame]['seq_id']   = scene_name
                sds.append(scene[frame])

    print("dataset", dataset)
    print("total number of sample data:", len(sds))


    from multiprocessing import Pool
    with Pool(args.num_workers) as p:
        results = list(tqdm(p.imap(process, sds), total=len(sds)))
