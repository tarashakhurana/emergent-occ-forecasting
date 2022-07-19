#!/usr/bin/env python3
import os
import json
import argparse
from re import T

import torch
from torch import nn
from torch.utils.data import DataLoader

from nuscenes.nuscenes import NuScenes

from data import nuScenesDataset, CollateFn
from model import *

import matplotlib.pyplot as plt
import numpy as np

from skimage.draw import polygon

def make_data_loader(cfg, args):
    if "train_on_all_sweeps" not in cfg:
        train_on_all_sweeps = False
    else:
        train_on_all_sweeps = cfg["train_on_all_sweeps"]

    dataset_kwargs = {
        "n_input": cfg["n_input"],
        "n_samples": args.n_samples,
        "n_output": cfg["n_output"],
        "scene_token": args.plot_scene,
        "train_on_all_sweeps": train_on_all_sweeps
    }
    data_loader_kwargs = {
        "pin_memory": False,  # NOTE
        "shuffle": False,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers
    }
    nusc = NuScenes(cfg["nusc_version"], cfg["nusc_root"])
    data_loader = DataLoader(nuScenesDataset(nusc, args.plot_split, dataset_kwargs),
                             collate_fn=CollateFn, **data_loader_kwargs)
    return data_loader

def mkdir_if_not_exists(d):
    if not os.path.exists(d):
        print(f"creating directory {d}")
        os.makedirs(d)

def voxelize_point_cloud(points):
    valid = (points[:, -1] == 0)
    x, y, z, t = points[valid].T
    x = ((x + 40.0) / 0.2).astype(int)
    y = ((y + 70.4) / 0.2).astype(int)
    mask = np.logical_and(
        np.logical_and(0 <= x, x < 400),
        np.logical_and(0 <= y, y < 704)
    )
    voxel_map = np.zeros((704, 400), dtype=bool)
    voxel_map[y[mask], x[mask]] = True
    return voxel_map

def normalize_cost_maps(cost_maps):
    batch_size = len(cost_maps)
    for i in range(batch_size):
        T = len(cost_maps[i])
        for t in range(T):
            cost_map = cost_maps[i, t]
            cost_min, cost_max = cost_map.min(), cost_map.max()
            cost_maps[i, t] = (cost_map - cost_min) / (cost_max - cost_min)
    return cost_maps

def embed_plans(imgs, plans, color, radius=1):
    # costs: N x T x H x W
    # plans: N x T x 3 (x, y, theta)
    assert(len(imgs) == len(plans))
    batch_size = len(imgs)
    half_r = int(radius // 2)
    for i in range(batch_size):
        T = len(imgs[i])
        xx = plans[i, :, 0]
        yy = plans[i, :, 1]
        xxi = ((xx + 40.0) / 0.2).astype(int)
        yyi = ((yy + 70.4) / 0.2).astype(int)
        for t in range(T):
            xi, yi = xxi[t], yyi[t]
            imgs[i, t, yi-radius:yi+radius+1, xi-radius:xi+radius+1, :] = color
            imgs[i, :, yi-half_r:yi+half_r+1, xi-half_r:xi+half_r+1, :] = color
    return imgs

def voxelize(pts, reverse_time=False):
    B = len(pts)
    T = int(pts[:, :, 3].max()) + 1

    bi = np.tile(np.arange(B)[:, None], (1, pts.shape[1]))
    xi = ((pts[:, :, 0] + 40.0) / 0.2).astype(int)
    yi = ((pts[:, :, 1] + 70.4) / 0.2).astype(int)
    ti = pts[:, :, 3].astype(int)
    mask = np.logical_and(
        ti >= 0,
        np.logical_and(
            np.logical_and(0 <= xi, xi < 400),
            np.logical_and(0 <= yi, yi < 704)
        )
    )

    bi = bi[mask]
    ti = ti[mask]
    if reverse_time:
        ti = T - 1 - ti
    yi = yi[mask]
    xi = xi[mask]

    imgs = np.zeros((len(pts), T, 704, 400, 3))
    imgs[bi, ti, yi, xi, :] = 1

    return imgs

def flip(img):
    return np.transpose(img, (1, 0, 2))

def plot(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_count = torch.cuda.device_count()
    if args.batch_size % device_count != 0:
        raise RuntimeError(f"Batch size ({args.batch_size}) cannot be divided by device count ({device_count})")

    model_dir = args.model_dir
    with open(f"{model_dir}/config.json", 'r') as f:
        cfg = json.load(f)

    # dataset
    data_loader = make_data_loader(cfg, args)

    # instantiate a model and a renderer
    _n_input, _n_output = cfg["n_input"], cfg["n_output"]
    _pc_range, _voxel_size = cfg["pc_range"], cfg["voxel_size"]

    model_type = cfg["model_type"]
    if model_type == "vanilla":
        model = VanillaNeuralMotionPlanner(_n_input, _n_output, _pc_range, _voxel_size)
    elif model_type == "vf_guided":
        model = VFGuidedNeuralMotionPlanner(_n_input, _n_output, _pc_range, _voxel_size)
    elif model_type == "obj_guided":
        model = ObjGuidedNeuralMotionPlanner(_n_input, _n_output, _pc_range, _voxel_size)
    elif model_type == "obj_shadow_guided":
        model = ObjShadowGuidedNeuralMotionPlanner(_n_input, _n_output, _pc_range, _voxel_size)
    elif model_type == "vf_explicit":
        model = VFExplicitNeuralMotionPlanner(_n_input, _n_output, _pc_range, _voxel_size, cfg["nvf_loss_factor"])
    elif model_type == "obj_explicit":
        model = ObjExplicitNeuralMotionPlanner(_n_input, _n_output, _pc_range, _voxel_size, cfg["obj_loss_factor"])
    elif model_type == "occ_explicit":
        model = OccExplicitNeuralMotionPlanner(_n_input, _n_output, _pc_range, _voxel_size, cfg["occ_loss_factor"])
    elif model_type == "vf_supervised":
        model = VFSupervisedNeuralMotionPlanner(_n_input, _n_output, _pc_range, _voxel_size, cfg["nvf_loss_factor"])
    elif model_type == "obj_supervised":
        model = ObjSupervisedNeuralMotionPlanner(_n_input, _n_output, _pc_range, _voxel_size, cfg["obj_loss_factor"])
    elif model_type == "obj_shadow_supervised":
        model = ObjShadowSupervisedNeuralMotionPlanner(_n_input, _n_output, _pc_range, _voxel_size, cfg["obj_loss_factor"])
    elif model_type == "obj_supervised_raymax":
        model = ObjSupervisedRaymaxNeuralMotionPlanner(_n_input, _n_output, _pc_range, _voxel_size, cfg["obj_loss_factor"])
    elif model_type == "lat_occ":
        model = LatOccNeuralMotionPlanner(_n_input, _n_output, _pc_range, _voxel_size, cfg["occ_loss_factor"])
    elif model_type == "lat_occ_vf_supervised":
        model = LatOccVFSupervisedNeuralMotionPlanner(_n_input, _n_output, _pc_range, _voxel_size, cfg["nvf_loss_factor"])
    elif model_type == "lat_occ_flow_vf_supervised":
        model = LatOccFlowVFSupervisedNeuralMotionPlanner(_n_input, _n_output, _pc_range, _voxel_size, cfg["nvf_loss_factor"])
    elif model_type == "lat_occ_multiflow_vf_supervised":
        model = LatOccMultiFlowVFSupervisedNeuralMotionPlanner(_n_input, _n_output, _pc_range, _voxel_size, cfg["flow_mode"], cfg["nvf_loss_factor"])
    else:
        raise NotImplementedError(f"{model_type} not implemented yet.")

    model = model.to(device)

    # resume
    ckpt_path = f"{args.model_dir}/ckpts/model_epoch_{args.plot_epoch}.pth"
    checkpoint = torch.load(ckpt_path, map_location=device)
    # NOTE: ignore renderer's parameters
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    # data parallel
    model = nn.DataParallel(model)
    model.eval()

    # output
    vis_dir = os.path.join(model_dir, "videos", f"{args.plot_split}_epoch_{args.plot_epoch}")
    mkdir_if_not_exists(vis_dir)

    #
    RED = (1.0, 0.0, 0.0)
    GREEN = (0.0, 1.0, 0.0)
    BLUE = (0.0, 0.0, 1.0)

    #
    np.set_printoptions(suppress=True, precision=2)
    num_batch = len(data_loader)
    for i, batch in enumerate(data_loader):
        sample_data_tokens = batch["sample_data_tokens"]
        bs = len(sample_data_tokens)
        if bs < device_count:
            print(f"Dropping the last batch of size {bs}")
            continue

        with torch.set_grad_enabled(False):
            results = model(batch, "test")

        sampled_plans = batch["sampled_trajectories"].detach().cpu().numpy()
        # sampled_plans_fine = batch["sampled_trajectories_fine"].detach().cpu().numpy()
        gt_plans = batch["gt_trajectories"].detach().cpu().numpy()
        best_plans = results["best_plans"].detach().cpu().numpy()

        ii = np.arange(bs)
        pred_plans = sampled_plans[ii, best_plans[ii, 0]]

        # tt = list(range(_n_output))

        if "cost" in results:
            costs = normalize_cost_maps(results["cost"].detach().cpu().numpy())
            cost_imgs = np.stack((costs, costs, costs), axis=-1)
            cost_imgs = embed_plans(cost_imgs, pred_plans, color=BLUE, radius=4)
        else:
            cost_imgs = None

        if "il_cost" in results:
            il_costs = normalize_cost_maps(results["il_cost"].detach().cpu().numpy())
            il_imgs = np.stack((il_costs, il_costs, il_costs), axis=-1)
            # il_imgs = embed_plans(il_imgs, gt_plans, color=GREEN, radius=4)
            il_imgs = embed_plans(il_imgs, pred_plans, color=BLUE, radius=4)
        else:
            il_imgs = None

        if "obj_prob" in results:
            obj_prob = results["obj_prob"].detach().cpu().numpy()
            _zeros = np.zeros_like(obj_prob)
            obj_imgs = np.stack((obj_prob, _zeros, _zeros), axis=-1)
        else:
            obj_imgs = None

        if "obj_cost" in results:
            obj_costs = normalize_cost_maps(results["obj_cost"].detach().cpu().numpy())
            obj_cost_imgs = np.stack((obj_costs, obj_costs, obj_costs), axis=-1)
            obj_cost_imgs = embed_plans(obj_cost_imgs, pred_plans, color=BLUE, radius=4)
        else:
            obj_cost_imgs = None

        if "occ_prob" in results:
            occ_prob = results["occ_prob"].detach().cpu().numpy()
            _zeros = np.zeros_like(occ_prob)
            occ_imgs = np.stack((occ_prob, _zeros, _zeros), axis=-1)
            occ_imgs = embed_plans(occ_imgs, pred_plans, color=BLUE, radius=4)
        else:
            occ_imgs = None

        if "occ_cost" in results:
            occ_costs = normalize_cost_maps(results["occ_cost"].detach().cpu().numpy())
            occ_cost_imgs = np.stack((occ_costs, occ_costs, occ_costs), axis=-1)
            occ_cost_imgs = embed_plans(occ_cost_imgs, pred_plans, color=BLUE, radius=4)
        else:
            occ_cost_imgs = None

        if "nvf_prob" in results:
            nvf_prob = results["nvf_prob"].detach().cpu().numpy()
            vf_prob = 1 - nvf_prob
            _zeros = np.zeros_like(vf_prob)
            vf_imgs = np.stack((_zeros, vf_prob, _zeros), axis=-1)
            vf_imgs = embed_plans(vf_imgs, pred_plans, color=BLUE, radius=4)
        else:
            vf_imgs = None

        if "nvf_cost" in results:
            nvf_costs = normalize_cost_maps(results["nvf_cost"].detach().cpu().numpy())
            nvf_cost_imgs = np.stack((nvf_costs, nvf_costs, nvf_costs), axis=-1)
            nvf_cost_imgs = embed_plans(nvf_cost_imgs, pred_plans, color=BLUE, radius=4)
        else:
            nvf_cost_imgs = None

        input_points = batch["input_points"].detach().cpu().numpy()
        input_imgs = voxelize(input_points, reverse_time=True)

        output_points = batch["output_points"].detach().cpu().numpy()
        output_imgs = voxelize(output_points)
        output_imgs = embed_plans(output_imgs, gt_plans, color=BLUE, radius=4)

        for j, sample_data_token in enumerate(sample_data_tokens):
            # visualization:
            # - highlight the low cost regions (sub-zero)
            # - distinguish cost maps from different timestamps

            # plot input sensor measurements
            for t in range(input_imgs.shape[1]):
                plt.imsave(f"{vis_dir}/{sample_data_token}_input_{t:02}.png", flip(input_imgs[j, t]))

            # plot future sensor measurements
            for t in range(output_imgs.shape[1]):
                plt.imsave(f"{vis_dir}/{sample_data_token}_output_{t:02}.png", flip(output_imgs[j, t]))

            # tt = [2, 4, 6]
            if cost_imgs is not None:
                for t in range(cost_imgs.shape[1]):
                    plt.imsave(f"{vis_dir}/{sample_data_token}_cost_{t:02}.png", flip(cost_imgs[j, t]))

            if il_imgs is not None:
                for t in range(il_imgs.shape[1]):
                    plt.imsave(f"{vis_dir}/{sample_data_token}_il_{t:02}.png", flip(il_imgs[j, t]))

            if obj_imgs is not None:
                for t in range(obj_imgs.shape[1]):
                    plt.imsave(f"{vis_dir}/{sample_data_token}_obj_{t:02}.png", flip(obj_imgs[j, t]))

            if obj_cost_imgs is not None:
                for t in range(obj_cost_imgs.shape[1]):
                    plt.imsave(f"{vis_dir}/{sample_data_token}_obj_cost_{t:02}.png", flip(obj_cost_imgs[j, t]))

            if nvf_cost_imgs is not None:
                for t in range(nvf_cost_imgs.shape[1]):
                    plt.imsave(f"{vis_dir}/{sample_data_token}_nvf_cost_{t:02}.png", flip(nvf_cost_imgs[j, t]))

            if occ_cost_imgs is not None:
                for t in range(occ_cost_imgs.shape[1]):
                    plt.imsave(f"{vis_dir}/{sample_data_token}_occ_cost_{t:02}.png", flip(occ_cost_imgs[j, t]))

            # if occ_costs is not None:
            #     occ_cost = np.concatenate(occ_costs[j, tt], axis=-1)
            #     # plt.imsave(f"{vis_dir}/{sample_data_token}_occ.png", occ_cost[::-1], cmap="gray")
            #     plt.imsave(f"{vis_dir}/{sample_data_token}_env.png", occ_cost[::-1], cmap="gray")

            # if nvf_costs is not None:
            #     nvf_cost = np.concatenate(nvf_costs[j, tt], axis=-1)
            #     # plt.imsave(f"{vis_dir}/{sample_data_token}_nvf.png", nvf_cost[::-1], cmap="gray")
            #     plt.imsave(f"{vis_dir}/{sample_data_token}_env.png", nvf_cost[::-1], cmap="gray")

            # if obj_costs is not None:
            #     obj_cost = np.concatenate(obj_costs[j, tt], axis=-1)
            #     # plt.imsave(f"{vis_dir}/{sample_data_token}_obj.png", obj_cost[::-1], cmap="gray")
            #     plt.imsave(f"{vis_dir}/{sample_data_token}_env.png", obj_cost[::-1], cmap="gray")

            # if costs is not None:
            #     cost = np.concatenate(costs[j, tt], axis=-1)
            #     plt.imsave(f"{vis_dir}/{sample_data_token}.png", cost[::-1], cmap="gray")

            if occ_imgs is not None:
                for t in range(occ_imgs.shape[1]):
                    plt.imsave(f"{vis_dir}/{sample_data_token}_occ_{t:02}.png", flip(occ_imgs[j, t]))

            if vf_imgs is not None:
                for t in range(vf_imgs.shape[1]):
                    plt.imsave(f"{vis_dir}/{sample_data_token}_vf_{t:02}.png", flip(vf_imgs[j, t]))

        print(f"{args.plot_split} Epoch-{args.plot_epoch},",
              f"Batch: {i+1}/{num_batch},")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--plot-split", type=str, required=True)
    parser.add_argument("--plot-epoch", type=int, default=5)
    parser.add_argument("--plot-scene", type=str, default="")
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=36)
    parser.add_argument("--cache-on", action="store_true")
    parser.add_argument("--cache-every", type=int, default=1)
    parser.add_argument("--plot-on", action="store_true")
    parser.add_argument("--plot-every", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=18)

    args = parser.parse_args()

    np.random.seed(0)
    torch.random.manual_seed(0)

    plot(args)


# #!/usr/bin/env python3
# import os
# import json
# import imageio
# import argparse

# import numpy as np

# import torch

# from torch import nn
# from torch.utils.data import DataLoader

# import matplotlib.pyplot as plt

# from nuscenes.nuscenes import NuScenes

# from data import nuScenesDataset, CollateFn
# from model import OccupancyForecastingNetwork

# def make_data_loader(cfg, args):
#     dataset_kwargs = {
#         "n_input": cfg["n_input"],
#         "n_output": cfg["n_output"],
#     }
#     data_loader_kwargs = {
#         "pin_memory": False,  # NOTE
#         "shuffle": True,
#         "batch_size": args.batch_size,
#         "num_workers": args.num_workers
#     }

#     nusc = NuScenes(cfg["nusc_version"], cfg["nusc_root"])
#     data_loader = DataLoader(nuScenesDataset(nusc, args.plot_split, dataset_kwargs),
#                              collate_fn=CollateFn, **data_loader_kwargs)
#     return data_loader

# def mkdir_if_not_exists(d):
#     if not os.path.exists(d):
#         print(f"creating directory {d}")
#         os.makedirs(d)

# def plot(args):
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#     #
#     device_count = torch.cuda.device_count()

#     #
#     model_dir = args.model_dir
#     with open(f"{model_dir}/config.json", 'r') as f:
#         cfg = json.load(f)

#     assert(model_dir == cfg["model_dir"])

#     # dataset
#     data_loader = make_data_loader(cfg, args)

#     # instantiate a model and a renderer
#     _n_input, _n_output = cfg["n_input"], cfg["n_output"]
#     _pc_range, _voxel_size = cfg["pc_range"], cfg["voxel_size"]
#     model = OccupancyForecastingNetwork(_n_input, _n_output, _pc_range, _voxel_size)
#     model = model.to(device)

#     # resume
#     ckpt_path = f"{args.model_dir}/ckpts/model_epoch_{args.plot_epoch}.pth"
#     checkpoint = torch.load(ckpt_path, map_location=device)

#     # NOTE: ignore renderer's parameters
#     model.load_state_dict(checkpoint["model_state_dict"], strict=False)

#     # data parallel
#     model = nn.DataParallel(model)
#     model.eval()

#     #
#     visual_dir = f"{args.model_dir}/visuals_{args.plot_split}/epoch_{args.plot_epoch}"

#     #
#     mkdir_if_not_exists(visual_dir)

#     #
#     for i, batch in enumerate(data_loader):
#         filenames = batch[0]
#         input_points, output_origins, output_points = batch[1:4]

#         bs = len(input_points)
#         if bs < device_count:
#             print(f"Dropping the last batch of size {bs}")
#             continue

#         with torch.set_grad_enabled(False):
#             ret_dict = model(input_points, output_origins, output_points,
#                              return_pred=True, return_label=True)

#         pred = ret_dict["pred"].detach().cpu().numpy()
#         label = ret_dict["label"].detach().cpu().numpy()

#         for j in range(len(filenames)):
#             scene_token, _, sd_token = filenames[j]

#             pred_path = f"{visual_dir}/{scene_token}_{sd_token}_pred.gif"
#             with imageio.get_writer(pred_path, mode="I", duration=0.5) as writer:
#                 for t in range(_n_output):
#                     writer.append_data(pred[j][t][::-1, :])

#             label_path = f"{visual_dir}/{scene_token}_{sd_token}_label.gif"
#             with imageio.get_writer(label_path, mode="I", duration=0.5) as writer:
#                 for t in range(_n_output):
#                     writer.append_data((label[j][t][::-1, :]+1)/2)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()

#     parser.add_argument("--model-dir", type=str, required=True)
#     parser.add_argument("--plot-split", type=str, required=True)
#     parser.add_argument("--plot-epoch", type=int, required=True)
#     parser.add_argument("--batch-size", type=int, default=36)
#     parser.add_argument("--num-workers", type=int, default=18)

#     args = parser.parse_args()

#     torch.manual_seed(1)

#     plot(args)
