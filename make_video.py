# make_seq_video.py
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
plt.ion()

parser = argparse.ArgumentParser()
parser.add_argument("--nusc-root", type=str, default="/data/nuscenes")
parser.add_argument("--nusc-version", type=str, default="v1.0-trainval")
args = parser.parse_args()

nusc_version = args.nusc_version
nusc_root = args.nusc_root

nusc = NuScenes(nusc_version, nusc_root)

soi = [
    # "2eb0dd074d8e4a328fd2283184c4412e",
    "0e7ede02718341558414865d5c604745"
]

model_dir = "models/lat_occ_vf_supervised_nmp_10000"
model_epoch = 9

for scene in nusc.scene:
    if scene["token"] not in soi:
        continue

    scene_dir = f"videos_new/{scene['token']}"
    if not os.path.exists(scene_dir):
        os.makedirs(scene_dir)

    log_pt_dir = f"{scene_dir}/logs_pt"
    log_pt_obj_dir = f"{scene_dir}/logs_pt_obj"
    log_pt_obj_fvf_dir = f"{scene_dir}/logs_pt_obj_fvf"
    log_pt_occ_dir = f"{scene_dir}/logs_pt_occ"
    log_pt_cost_dir = f"{scene_dir}/logs_pt_cost"
    log_pt_occ_cost_dir = f"{scene_dir}/logs_pt_occ_cost"
    for d in [log_pt_dir, log_pt_obj_dir, log_pt_obj_fvf_dir, log_pt_occ_dir, log_pt_cost_dir, log_pt_occ_cost_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    first_sample = nusc.get("sample", scene["first_sample_token"])
    sd_token = first_sample["data"]["LIDAR_TOP"]
    count = 0
    while sd_token != "":
        fvf_path = f"/data/nuscenes/fvfmaps/{nusc_version}/{sd_token}.bin"
        obj_path = f"/data/nuscenes/obj_boxes/{nusc_version}/{sd_token}.bin"
        grnd_path = f"/data/nuscenes/grndseg/{nusc_version}/{sd_token}_grndseg.bin"

        occ_path = f"{model_dir}/videos/val_epoch_{model_epoch}/{sd_token}_occ_00.png"
        cost_path = f"{model_dir}/videos/val_epoch_{model_epoch}/{sd_token}_cost_00.png"
        cost_paths = [f"{model_dir}/videos/val_epoch_{model_epoch}/{sd_token}_cost_{i:02}.png" for i in range(7)]
        occ_cost_path = f"{model_dir}/videos/val_epoch_{model_epoch}/{sd_token}_occ_cost_00.png"
        occ_cost_paths = [f"{model_dir}/videos/val_epoch_{model_epoch}/{sd_token}_occ_cost_{i:02}.png" for i in range(7)]

        sd = nusc.get("sample_data", sd_token)

        if os.path.exists(grnd_path) and os.path.exists(fvf_path) and os.path.exists(obj_path):
            gseg = np.fromfile(grnd_path, dtype=np.int8)

            pc = LidarPointCloud.from_file(f"{nusc.dataroot}/{sd['filename']}")
            sample = nusc.get("sample", sd["sample_token"])
            scene = nusc.get("scene", sample["scene_token"])
            log = nusc.get("log", scene["log_token"])
            flip_flag = True if log["location"].startswith("singapore") else False

            if flip_flag:
                pc.points[0] *= -1

            X, Y, Z, _ = pc.points
            pt_map = np.zeros((704, 400))
            Xi = ((X + 40.0) / 0.2).astype(int)
            Yi = ((Y + 70.4) / 0.2).astype(int)
            mask = np.logical_and(np.logical_and(0 <= Xi, Xi < 400),
                                  np.logical_and(0 <= Yi, Yi < 704))
            Yi1, Xi1 = Yi[mask], Xi[mask]
            pt_map[Yi1, Xi1] = 1

            mask2 = np.logical_and(mask, np.logical_and(24 <= gseg, gseg <= 27))
            Yi2, Xi2 = Yi[mask2], Xi[mask2]
            pt_map[Yi2, Xi2] = 0.2

            # img_pt_only = img.copy()
            # img_pt_only[:, :, [0, 2]] = 1
            pt_img = np.dstack((pt_map, pt_map, pt_map))
            # plt.imsave(f"{log_pt_dir}/{count:04}.png", pt_img[::-1])
            plt.imsave(f"{log_pt_dir}/{count:04}.png", np.transpose(pt_img, (1, 0, 2)))
            
            obj_maps = np.fromfile(obj_path, dtype=bool)
            obj_maps = obj_maps.reshape((-1, 704, 400))

            _zeros = np.zeros_like(obj_maps)
            obj_imgs = np.stack((obj_maps, _zeros, _zeros), axis=-1)
            # obj_img = np.where(obj_maps[0, :, :, None], obj_imgs[0], pt_img)
            # obj_img = obj_imgs[0] / 2 + pt_img / 2
            pt_obj_img = obj_imgs[0] / 2 + pt_img / 2
            obj_img = np.where(obj_maps[0, :, :, None], pt_obj_img, pt_img)
            # obj_img = np.where(obj_maps[0, :, :, None], obj_imgs[0], pt_img)
            # plt.imsave(f"{log_pt_obj_dir}/{count:04}.png", obj_img[::-1])
            plt.imsave(f"{log_pt_obj_dir}/{count:04}.png", np.transpose(obj_img, (1, 0, 2)))

            fvf_maps = np.fromfile(fvf_path, dtype=np.int8)
            fvf_maps = fvf_maps.reshape((-1, 704, 400))
            fvf_maps = (fvf_maps == -1)

            fvf_imgs = np.stack((_zeros, fvf_maps, _zeros), axis=-1)
            obj_fvf_img = fvf_imgs[0] / 2 + obj_img / 2
            fvf_img = np.where(fvf_maps[0, :, :, None], obj_fvf_img, obj_img)
            
            # img = (1/3) * pt_map + (1/3) * fvf_maps[0] + (1/3) * obj_maps[0]
            # img = np.dstack((obj_maps[0], pt_map, fvf_maps[0]))
            # img
            plt.imsave(f"{log_pt_obj_fvf_dir}/{count:04}.png", np.transpose(fvf_img, (1, 0, 2)))

            if os.path.exists(occ_path):
                occ_img = np.transpose(plt.imread(occ_path)[:, :, :3], (1, 0, 2))
                pt_occ_img = occ_img / 2 + pt_img / 2
                alpha = occ_img[:, :, 0, None]
                occ_img = alpha * pt_occ_img + (1 - alpha) * pt_img
                plt.imsave(f"{log_pt_occ_dir}/{count:04}.png", np.transpose(occ_img, (1, 0, 2)))

            if os.path.exists(cost_path): 
                cost_imgs = [np.transpose(plt.imread(cost_path)[:, :, :3], (1, 0, 2)) for cost_path in cost_paths]
                cost_img = np.max(np.stack(cost_imgs), 0)
                pt_cost_img = cost_img / 2 + pt_img / 2
                alpha = cost_img[:, :, 0, None]
                cost_img = alpha * pt_cost_img + (1 - alpha) * pt_img
                plt.imsave(f"{log_pt_cost_dir}/{count:04}.png", np.transpose(cost_img, (1, 0, 2)))

            if os.path.exists(occ_cost_path):
                occ_cost_imgs = [np.transpose(plt.imread(occ_cost_path)[:, :, :3], (1, 0, 2)) for occ_cost_path in occ_cost_paths]
                occ_cost_img = np.max(np.stack(occ_cost_imgs), 0)
                # pt_occ_cost_img = occ_cost_img / 2 + pt_img / 2
                alpha = np.max(occ_cost_img, axis=-1, keepdims=True)
                occ_cost_img = alpha * occ_cost_img + (1 - alpha) * pt_img
                occ_cost_img = np.maximum(0.0, np.minimum(1.0, occ_cost_img))
                plt.imsave(f"{log_pt_occ_cost_dir}/{count:04}.png", np.transpose(occ_cost_img, (1, 0, 2)))

            # frame_dir = f"{scene_dir}/frames/{sd_token}"
            # if not os.path.exists(frame_dir):
            #     os.makedirs(frame_dir)

            # for t in range(len(fvf_maps)):
            #     plt.imsave(f"{frame_dir}/{count:04}.png", fvf_maps[t])

            count += 1
        sd_token = sd["next"]
