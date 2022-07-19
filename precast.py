# precast.py
# raycasting and cache all future visible freespace
# Usage:
#   python -W ignore precast.py

import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes
from util.once_devkit.once import ONCE
from data import nuScenesDataset, ONCEDataset, CollateFn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.cpp_extension import load

torch.random.manual_seed(0)
np.random.seed(0)

dataset = "nusc"

renderer = load(
    "renderer",
    sources=["lib/render/renderer.cpp", "lib/render/renderer.cu"],
    verbose=True,
)
raycaster = load(
    "raycaster",
    sources=["lib/raycast/raycaster.cpp", "lib/raycast/raycaster.cu"],
    verbose=True,
)

# nusc = NuScenes("v1.0-mini", "/data/nuscenes")
if dataset == "nusc":
    nusc = NuScenes("v1.0-trainval", "/data3/tkhurana/datasets/nuScenes")
elif dataset == "once":
    once = ONCE("/data3/tkhurana/datasets/once")

# dataset_kwargs = {"n_input": 20, "n_samples": 100, "n_output": 7}
nusc_dataset_kwargs = {
    "n_input": 20,
    "n_samples": 100,
    "n_output": 7,
    "train_on_all_sweeps": True,
}
once_dataset_kwargs = {
    "n_input": 5,
    "n_samples": 100,
    "n_output": 7,
    "train_on_all_sweeps": True,
    "sampled_trajectories": "curves",
    "sample_set": "",
}

if dataset == "nusc":
    # Dataset = nuScenesDataset(nusc, "train", nusc_dataset_kwargs)
    Dataset = nuScenesDataset(nusc, "val", nusc_dataset_kwargs)
elif dataset == "once":
    Dataset = ONCEDataset(once, "val", once_dataset_kwargs)

data_loader_kwargs = {
    "pin_memory": False,
    "shuffle": False,
    "batch_size": 32,
    "num_workers": 4,
}
data_loader = DataLoader(Dataset, collate_fn=CollateFn, **data_loader_kwargs)

pc_range = [-40.0, -70.4, -2.0, 40.0, 70.4, 3.4]
voxel_size = 0.2
output_grid = [7, 704, 400]
input_grid = [7, 704, 400]

device = torch.device("cuda:0")

offset = torch.nn.parameter.Parameter(
    torch.Tensor(pc_range[:3])[None, None, :], requires_grad=False
).to(device)
scaler = torch.nn.parameter.Parameter(
    torch.Tensor([voxel_size] * 3)[None, None, :], requires_grad=False
).to(device)

if dataset == "nusc":
    cache_dir = f"{nusc.dataroot}/fvfmaps/{nusc.version}"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
elif dataset == "once":
    cache_dir = f"{once.data_root}"

for i, batch in enumerate(data_loader):
    print(i, len(data_loader))

    # if i != 0:
    #     continue

    sd_tokens = batch["sample_data_tokens"]
    """
    done = True
    for j, sd_token in enumerate(sd_tokens):

        if dataset == "nusc":
            path = f"{cache_dir}/fvfmaps/{sd_token}.bin"
        elif dataset == "once":
            path = f"{cache_dir}/{sd_token[0]}/fvfmaps/{sd_token[1]}.bin"

        if not os.path.exists(path):
            done = False
            break

    if done:
        continue
    """

    output_origins = batch["output_origins"].to(device)
    output_points = batch["output_points"].to(device)

    print(
        "shape of the ray origins and points", output_origins.shape, output_points.shape
    )

    output_origins[:, :, :3] = (output_origins[:, :, :3] - offset) / scaler
    output_points[:, :, :3] = (output_points[:, :, :3] - offset) / scaler

    # what we would like
    freespace = raycaster.raycast(output_origins, output_points, output_grid)
    freespace = freespace.detach().cpu().numpy().astype(np.int8)

    offset_ = torch.nn.parameter.Parameter(
        torch.Tensor(pc_range[:2])[None, None, :], requires_grad=False
    ).numpy()
    scaler_ = torch.nn.parameter.Parameter(
        torch.Tensor([voxel_size] * 2)[None, None, :], requires_grad=False
    ).numpy()

    gt_trajectory = batch["gt_trajectories"]
    print()
    gt_trajectory[:, :, :2] = (gt_trajectory[:, :, :2] - offset_) / scaler_
    output_origins = output_origins.cpu().numpy().astype(int)
    gt_trajectory = gt_trajectory.numpy()  # .astype(int)

    #
    for j, sd_token in enumerate(sd_tokens):
        # fvf: future visible freespace
        if dataset == "nusc":
            path = f"{cache_dir}/{sd_token}.bin"
            ptsdir = cache_dir.replace("fvfmaps", "lesspoints")
            os.makedirs(ptsdir, exist_ok=True)
            ptspath = f"{ptsdir}/{sd_token}.npy"
        elif dataset == "once":
            pathdir = f"{cache_dir}/{sd_token[0]}/fvfmaps/"
            ptsdir = f"{cache_dir}/{sd_token[0]}/lesspoints/"
            os.makedirs(pathdir, exist_ok=True)
            os.makedirs(ptsdir, exist_ok=True)
            imgpathdir = f"{pathdir}/imgs/"
            os.makedirs(imgpathdir, exist_ok=True)
            path = f"{cache_dir}/{sd_token[0]}/fvfmaps/{sd_token[1]}.bin"
            ptspath = f"{cache_dir}/{sd_token[0]}/lesspoints/{sd_token[1]}.npy"
            imgpath = f"{imgpathdir}/{sd_token[1]}.jpg"

        # if Dataset.data_split == "val":
        #     raise RuntimeError("This is unexpected!")

        # freespace_plot = np.zeros((704, 400 * 7))
        # for m in range(7):
        #     freespace_plot[:, m * 400: (m+1) * 400] = freespace[j].copy()[m].squeeze()
        freespace_plot = np.where(freespace[j] == 1, 0, freespace[j])
        freespace_plot = np.sum(freespace[j] + 1, axis=0)
        freespace_contour = np.where(freespace[j] == 1, 1, 0)
        for k in range(freespace_contour.shape[0]):
            indicesk = np.where(freespace_contour[k] == 1)
            cc = np.hstack(
                [
                    np.expand_dims(indicesk[1], -1),
                    np.expand_dims(indicesk[0], -1),
                    np.ones((indicesk[0].shape[0], 1)) + 1.0,
                ]
            )
            # print("cc shape", cc.shape)
            if k == 0:
                indices = np.expand_dims(cc, 0)
            else:
                if cc.shape[0] > indices.shape[1]:
                    shapefill = cc.shape[0] - indices.shape[1]
                    indices = np.concatenate(
                        (indices, np.full((indices.shape[0], shapefill, 3), -1)), axis=1
                    )
                else:
                    shapefill = indices.shape[1] - cc.shape[0]
                    cc = np.vstack([cc, np.full((shapefill, 3), -1)])
                # print("cc second shape", cc.shape, indices.shape)
                indices = np.vstack([indices, np.expand_dims(cc, axis=0)])
        # print(indices)
        less_points = indices
        # print("before", less_points, less_points.shape, scaler_.shape)
        less_points[..., :2] = less_points[..., :2] * scaler_ + offset_
        # print("after", less_points, less_points.shape)
        # for k in range(7):
        #     freespace_plot[gt_trajectory[j][k, 1], gt_trajectory[j][k, 0]] = 0 #  + output_origins[j][:, :2]] = 0

        # plt.rcParams["figure.figsize"] = (8, 14)
        # plt.imshow(freespace_plot, cmap='gray')
        # print("yaw angles", gt_trajectory[j][:, 2])
        # plt.plot(gt_trajectory[j][:, 0].astype(int), gt_trajectory[j][:, 1].astype(int))
        # plt.scatter(gt_trajectory[j][:, 0].astype(int), gt_trajectory[j][:, 1].astype(int), c='r')
        # plt.savefig(imgpath)
        # plt.close()
        # print("done with image", imgpath)
        print(less_points.shape)
        # less_points.tofile(ptspath)
        # np.save(ptspath, less_points)
        freespace[j].tofile(path)
    """



    input_origins = batch["input_origins"].to(device)
    input_points = batch["input_points"].to(device)

    print("input_+oriogins shae", input_origins.shape)

    input_origins[:, :, :3] = (input_origins[:, :, :3] - offset) / scaler
    input_points[:, :, :3] = (input_points[:, :, :3] - offset) / scaler

    # what we would like
    freespace = raycaster.raycast(input_origins, input_points, input_grid)
    freespace = freespace.detach().cpu().numpy().astype(np.int8)

    offset_ = torch.nn.parameter.Parameter(
        torch.Tensor(pc_range[:2])[None, None, :], requires_grad=False).numpy()
    scaler_ = torch.nn.parameter.Parameter(
        torch.Tensor([voxel_size]*2)[None, None, :], requires_grad=False).numpy()

    gt_trajectory = input_origins[:, :, :2].cpu().numpy().astype(int)
    print()
    input_origins = input_origins.cpu().numpy().astype(int)

    #
    for j, sd_token in enumerate(sd_tokens):
        # fvf: future visible freespace
        if dataset == "nusc":
            path = f"{cache_dir}/fvfmaps/{sd_token}.bin"
            imgpathdir = f"{cachedir}/imgs/"
            os.makedirs(imgpathdir, exist_ok=True)
            imgpath = f"{imgpathdir}/{sd_token}.jpg"
        elif dataset == "once":
            pathdir = f"{cache_dir}/{sd_token[0]}/fvfmaps/"
            os.makedirs(pathdir, exist_ok=True)
            imgpathdir = f"{pathdir}/imgs/"
            os.makedirs(imgpathdir, exist_ok=True)
            path = f"{cache_dir}/{sd_token[0]}/fvfmaps/{sd_token[1]}.bin"
            imgpath = f"{imgpathdir}/{sd_token[1]}_input.jpg"

        if Dataset.data_split == "val":
            raise RuntimeError("This is unexpected!")


        # freespace_plot = np.zeros((704, 400 * 7))
        # for m in range(7):
        #     freespace_plot[:, m * 400: (m+1) * 400] = freespace[j].copy()[m].squeeze()
        freespace_contour = np.where(freespace[j] == 1, 1, 0)
        freespace_plot = np.where(freespace[j] == 1, 0, freespace[j])
        freespace_plot = np.sum(freespace[j] + 1, axis=0)
        print(freespace_plot.shape, np.unique(freespace_plot))
        # for k in range(7):
        #     freespace_plot[gt_trajectory[j][k, 1], gt_trajectory[j][k, 0]] = 0 #  + output_origins[j][:, :2]] = 0

        plt.rcParams["figure.figsize"] = (8, 14)
        plt.imshow(freespace_plot, cmap='gray')
        plt.plot(gt_trajectory[j][:, 0].astype(int), gt_trajectory[j][:, 1].astype(int))
        plt.scatter(gt_trajectory[j][:, 0].astype(int), gt_trajectory[j][:, 1].astype(int), c='r')
        plt.savefig(imgpath)
        plt.close()

        plt.rcParams["figure.figsize"] = (8, 14)
        plt.imshow(freespace_contour, cmap='gray')
        plt.savefig(imgpath.replace('_input', '_contour'))
        plt.close()

        print("done with image", imgpath)
    """
