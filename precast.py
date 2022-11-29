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
    nusc = NuScenes("v1.0-trainval", "/data/nuScenes")
elif dataset == "once":
    once = ONCE("/data/once")

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
    Dataset = nuScenesDataset(nusc, "train", nusc_dataset_kwargs)
elif dataset == "once":
    Dataset = ONCEDataset(once, "train", once_dataset_kwargs)

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

    sd_tokens = batch["sample_data_tokens"]
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
            path = f"{pathdir}/{sd_token[1]}.bin"
            ptspath = f"{ptsdir}/{sd_token[1]}.npy"

        if Dataset.data_split == "val":
            raise RuntimeError("This is unexpected!")

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
                indices = np.vstack([indices, np.expand_dims(cc, axis=0)])
        less_points = indices
        less_points[..., :2] = less_points[..., :2] * scaler_ + offset_
        less_points.tofile(ptspath)
        # np.save(ptspath, less_points)
        freespace[j].tofile(path)
