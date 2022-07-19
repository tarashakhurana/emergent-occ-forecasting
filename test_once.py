#!/usr/bin/env python3
import os
import json
import argparse
from re import T

import sklearn
import torch
from torch import nn
from torch.utils.data import DataLoader

from nuscenes.nuscenes import NuScenes
from util.once_devkit.once import ONCE

from data import ONCEDataset, nuScenesDataset, CollateFn

# from model import OccupancyForecastingNetwork
# from model import NeuralMotionPlanner

from model import *

import matplotlib.pyplot as plt
import numpy as np
import cv2
import json

from skimage.draw import polygon
from torch.utils.cpp_extension import load
from scipy import ndimage

renderer = load(
    "renderer",
    sources=["lib/render/renderer.cpp", "lib/render/renderer.cu"],
    verbose=True,
)


def make_data_loader(cfg, args):
    if "train_on_all_sweeps" not in cfg:
        train_on_all_sweeps = False
    else:
        train_on_all_sweeps = cfg["train_on_all_sweeps"]
    dataset_kwargs = {
        "n_input": cfg["n_input"],
        "n_samples": args.n_samples,
        "n_output": cfg["n_output"],
        "train_on_all_sweeps": train_on_all_sweeps,
        "sampled_trajectories": args.sampled_trajectories,
        "sample_set": args.sample_set,
    }
    data_loader_kwargs = {
        "pin_memory": False,  # NOTE
        "shuffle": True,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
    }
    once = ONCE(cfg["data_root"])
    data_loader = DataLoader(
        ONCEDataset(once, args.test_split, dataset_kwargs),
        collate_fn=CollateFn,
        **data_loader_kwargs,
    )
    return data_loader


def mkdir_if_not_exists(d):
    if not os.path.exists(d):
        print(f"creating directory {d}")
        os.makedirs(d)


def voxelize_point_cloud(points):
    valid = points[:, -1] == 0
    x, y, z, t = points[valid].T
    x = ((x + 40.0) / 0.2).astype(int)
    y = ((y + 70.4) / 0.2).astype(int)
    mask = np.logical_and(
        np.logical_and(0 <= x, x < 400), np.logical_and(0 <= y, y < 704)
    )
    voxel_map = np.zeros((704, 400), dtype=bool)
    voxel_map[y[mask], x[mask]] = True
    return voxel_map


def rotate(img, text=""):
    img = ndimage.rotate(img, 270)
    text_color = (0, 255, 255)
    if text == "total cost + output trajectory":
        text_color = (0, 200, 255)
    img = cv2.putText(
        img, text, (10, 40), cv2.FONT_HERSHEY_PLAIN, 2, text_color, 2, cv2.LINE_AA
    )
    return img


def evaluate_box_coll(obj_boxes, trajectory, pc_range):
    xmin, ymin, _, xmax, ymax, _ = pc_range
    T, H, W = obj_boxes.shape
    collisions = np.full(T, False)
    for t in range(T):
        x, y, theta = trajectory[t]
        corners = np.array(
            [
                (-0.8, -1.5, 1),  # back left corner
                (0.8, -1.5, 1),  # back right corner
                (0.8, 2.5, 1),  # front right corner
                (-0.8, 2.5, 1),  # front left corner
            ]
        )
        tf = np.array(
            [
                [np.cos(theta), -np.sin(theta), x],
                [np.sin(theta), np.cos(theta), y],
                [0, 0, 1],
            ]
        )
        xx, yy = tf.dot(corners.T)[:2]

        yi = np.round((yy - ymin) / (ymax - ymin) * H).astype(int)
        xi = np.round((xx - xmin) / (xmax - xmin) * W).astype(int)
        rr, cc = polygon(yi, xi)
        I = np.logical_and(
            np.logical_and(rr >= 0, rr < H), np.logical_and(cc >= 0, cc < W),
        )
        collisions[t] = np.any(obj_boxes[t, rr[I], cc[I]])
    return collisions


def evaluate_obj_recall(obj_boxes, occ_prob):
    thresh = 0.5
    bin_occ = np.where(occ_prob > thresh, True, False)
    true_positives = np.count_nonzero(bin_occ & obj_boxes, axis=(1, 2))
    false_negatives = np.count_nonzero(obj_boxes & ~bin_occ, axis=(1, 2))
    recall = np.where(
        (false_negatives == 0) & (true_positives == 0),
        0,
        true_positives / (true_positives + false_negatives),
    )
    return recall


def make_cost_fig(cost_maps):
    cost_imgs = np.ones_like(cost_maps)
    T = len(cost_maps)
    for t in range(T):
        cost_map = cost_maps[t]
        cost_min, cost_max = cost_map.min(), cost_map.max()
        cost_img = (cost_map - cost_min) / (cost_max - cost_min)
        cost_imgs[t] = cost_img
    return cost_imgs


def test(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_count = torch.cuda.device_count()
    if args.batch_size % device_count != 0:
        raise RuntimeError(
            f"Batch size ({args.batch_size}) cannot be divided by device count ({device_count})"
        )

    print("Doing model:", args.model_dir)
    model_dir = args.model_dir
    with open(f"{model_dir}/config.json", "r") as f:
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
        model = ObjGuidedNeuralMotionPlanner(
            _n_input, _n_output, _pc_range, _voxel_size
        )
    elif model_type == "obj_shadow_guided":
        model = ObjShadowGuidedNeuralMotionPlanner(
            _n_input, _n_output, _pc_range, _voxel_size
        )
    elif model_type == "vf_explicit":
        model = VFExplicitNeuralMotionPlanner(
            _n_input, _n_output, _pc_range, _voxel_size, cfg["nvf_loss_factor"]
        )
    elif model_type == "obj_explicit":
        model = ObjExplicitNeuralMotionPlanner(
            _n_input, _n_output, _pc_range, _voxel_size, cfg["obj_loss_factor"]
        )
    elif model_type == "occ_explicit":
        model = OccExplicitNeuralMotionPlanner(
            _n_input, _n_output, _pc_range, _voxel_size, cfg["occ_loss_factor"]
        )
    elif model_type == "vf_supervised":
        model = VFSupervisedNeuralMotionPlanner(
            _n_input, _n_output, _pc_range, _voxel_size, cfg["nvf_loss_factor"]
        )
    elif model_type == "obj_supervised":
        model = ObjSupervisedNeuralMotionPlanner(
            _n_input, _n_output, _pc_range, _voxel_size, cfg["obj_loss_factor"]
        )
    elif model_type == "obj_shadow_supervised":
        model = ObjShadowSupervisedNeuralMotionPlanner(
            _n_input, _n_output, _pc_range, _voxel_size, cfg["obj_loss_factor"]
        )
    elif model_type == "obj_supervised_raymax":
        model = ObjSupervisedRaymaxNeuralMotionPlanner(
            _n_input, _n_output, _pc_range, _voxel_size, cfg["obj_loss_factor"]
        )
    elif model_type == "lat_occ":
        model = LatOccNeuralMotionPlanner(
            _n_input, _n_output, _pc_range, _voxel_size, cfg["occ_loss_factor"]
        )
    elif model_type == "lat_occ_vf_supervised":
        model = LatOccVFSupervisedNeuralMotionPlanner(
            _n_input,
            _n_output,
            _pc_range,
            _voxel_size,
            cfg["nvf_loss_factor"],
            args.dilate,
        )
    elif model_type == "lat_occ_vf_supervised_lat_occ":
        model = LatOccVFSupervisedLatOccNeuralMotionPlanner(
            _n_input, _n_output, _pc_range, _voxel_size, cfg["nvf_loss_factor"]
        )
    elif model_type == "lat_occ_vf_supervised_l2_costmargin":
        model = LatOccVFSupervisedL2CostMarginNeuralMotionPlanner(
            _n_input, _n_output, _pc_range, _voxel_size, cfg["nvf_loss_factor"]
        )
    elif model_type == "lat_occ_vf_supervised_xl":
        model = LatOccVFSupervisedXLNeuralMotionPlanner(
            _n_input, _n_output, _pc_range, _voxel_size, cfg["nvf_loss_factor"]
        )
    elif model_type == "lat_occ_vf_supervised_xxl":
        model = LatOccVFSupervisedXXLNeuralMotionPlanner(
            _n_input, _n_output, _pc_range, _voxel_size, cfg["nvf_loss_factor"]
        )
    elif model_type == "lat_occ_vf_supervised_xxxl":
        model = LatOccVFSupervisedXXXLNeuralMotionPlanner(
            _n_input, _n_output, _pc_range, _voxel_size, cfg["nvf_loss_factor"]
        )
    elif model_type == "lat_occ_flow_vf_supervised":
        model = LatOccFlowVFSupervisedNeuralMotionPlanner(
            _n_input, _n_output, _pc_range, _voxel_size, cfg["nvf_loss_factor"]
        )
    elif model_type == "lat_occ_multiflow_vf_supervised":
        model = LatOccMultiFlowVFSupervisedNeuralMotionPlanner(
            _n_input,
            _n_output,
            _pc_range,
            _voxel_size,
            cfg["flow_mode"],
            cfg["nvf_loss_factor"],
        )
    else:
        raise NotImplementedError(f"{model_type} not implemented yet.")

    model = model.to(device)

    # resume
    ckpt_path = f"{args.model_dir}/ckpts/model_epoch_{args.test_epoch}.pth"
    checkpoint = torch.load(ckpt_path, map_location=device)
    # NOTE: ignore renderer's parameters
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    # data parallel
    model = nn.DataParallel(model)
    model.eval()

    # output
    vis_dir = os.path.join(
        model_dir, "visuals_paper", f"{args.test_split}_epoch_{args.test_epoch}"
    )
    detailed_results_file = f"{model_dir}/detailed_results.json"
    dict_list = []
    # mkdir_if_not_exists(vis_dir)

    counts = np.zeros(cfg["n_output"], dtype=int)
    coll_counts = np.zeros(cfg["n_output"], dtype=int)
    l2_dist_sum = np.zeros(cfg["n_output"], dtype=float)
    obj_coll_sum = np.zeros(cfg["n_output"], dtype=int)
    obj_box_coll_sum = np.zeros(cfg["n_output"], dtype=int)
    obj_recall_sum = np.zeros(cfg["n_output"], dtype=float)
    if args.compute_dense_fvf_loss:
        dense_fvf_bce = np.zeros(cfg["n_output"], dtype=float)
        dense_fvf_f1 = np.zeros(cfg["n_output"], dtype=float)
        dense_fvf_ap = np.zeros(cfg["n_output"], dtype=float)
    if args.compute_raydist_loss:
        raydist_error = np.zeros(cfg["n_output"], dtype=float)

    #
    obj_box_dir = f"{cfg['data_root']}/obj_boxes/{cfg['data_version']}"

    #
    np.set_printoptions(suppress=True, precision=2)
    num_batch = len(data_loader)
    for i, batch in enumerate(data_loader):
        sample_data_tokens = batch["sample_data_tokens"]
        output_origins = batch["output_origins"]
        output_points = batch["output_points"]
        input_points = batch["input_points"]
        bs = len(sample_data_tokens)
        print(bs, device_count)
        if bs < device_count:
            print(f"Dropping the last batch of size {bs}")
            continue

        with torch.set_grad_enabled(False):
            results = model(batch, "test")

        best_plans = results["best_plans"].detach().cpu().numpy()
        if "occ_prob" in results:
            occ_probs = results["occ_prob"].detach().cpu()
        else:
            occ_probs = None

        sampled_plans = batch["sampled_trajectories"].detach().cpu().numpy()
        sampled_plans_fine = batch["sampled_trajectories_fine"].detach().cpu().numpy()
        gt_plans = batch["gt_trajectories"].detach().cpu().numpy()

        plot_on = args.plot_on and (i % args.plot_every == 0)
        render_freespace = args.render_freespace
        cache_on = args.cache_on and (i % args.cache_every == 0)

        if plot_on and "il_cost" in results:
            il_costs = results["il_cost"].detach().cpu().numpy()
        else:
            il_costs = None

        if plot_on and "occ_cost" in results:
            occ_costs = results["occ_cost"].detach().cpu().numpy()
        else:
            occ_costs = None

        if plot_on and "nvf_prob" in results:
            nvf_probs = results["nvf_prob"].detach().cpu().numpy()
        else:
            nvf_probs = None

        if plot_on and "nvf_cost" in results:
            nvf_costs = results["nvf_cost"].detach().cpu().numpy()
        else:
            nvf_costs = None

        if plot_on and "obj_cost" in results:
            obj_costs = results["obj_cost"].detach().cpu().numpy()
        else:
            obj_costs = None

        if (cache_on or plot_on) and "cost" in results:
            costs = results["cost"].detach().cpu().numpy()
        else:
            costs = None

        if args.compute_dense_fvf_loss:
            nvf_probs = results["nvf_prob"].detach().cpu().numpy().astype("float64")
            nvf_gts = batch["fvf_maps"].detach().cpu().numpy()
            print("unique values in nvf_gt", np.unique(nvf_gts))
            nvf_gts = np.where(nvf_gts == 1, 0, nvf_gts)
            nvf_gts += 1
            nvf_probs = 1 - nvf_probs
            nvf_gts = 1 - nvf_gts

        if args.compute_raydist_loss:
            nvf_probs = results["nvf_prob"].detach().cpu().numpy().astype("float64")

        whitebg = np.zeros((704, 400)) + 255
        whitebg = whitebg.astype(np.uint8)

        for j, sample_data_token in enumerate(sample_data_tokens):
            # visualization:
            # - highlight the low cost regions (sub-zero)
            # - distinguish cost maps from different timestamps
            # if sample_data_token[2] != 716 and sample_data_token[2] != 2078:
            #     continue
            # if sample_data_token[2] != 47 and sample_data_token[2] != 489 and sample_data_token[2] != 189:
            #     continue
            # if sample_data_token[2] != 1535 and sample_data_token[2] != 1554 and sample_data_token[2] != 55:
            #     continue
            # if sample_data_token[2] != 981: # 157: # 1312:
            #     continue
            # print(output_origins[j])
            # if (
            #     sample_data_token[2] != 390
            #     and sample_data_token[2] != 440
            #     and sample_data_token[2] != 132
            #     and sample_data_token[2] != 346
            #     and sample_data_token[2] != 164
            # ):
            #     continue
            vis_dir_seq = os.path.join(
                vis_dir, f"{sample_data_token[0]}/{sample_data_token[2]}/"
            )
            os.makedirs(vis_dir_seq, exist_ok=True)
            vis_dir_seq = os.path.join(
                vis_dir, f"{sample_data_token[0]}/{sample_data_token[2]}/il/"
            )
            os.makedirs(vis_dir_seq, exist_ok=True)
            vis_dir_seq = os.path.join(
                vis_dir, f"{sample_data_token[0]}/{sample_data_token[2]}/occ/"
            )
            os.makedirs(vis_dir_seq, exist_ok=True)
            vis_dir_seq = os.path.join(
                vis_dir, f"{sample_data_token[0]}/{sample_data_token[2]}/nvf/"
            )
            os.makedirs(vis_dir_seq, exist_ok=True)
            vis_dir_seq = os.path.join(
                vis_dir, f"{sample_data_token[0]}/{sample_data_token[2]}/nvfcost/"
            )
            os.makedirs(vis_dir_seq, exist_ok=True)
            vis_dir_seq = os.path.join(
                vis_dir, f"{sample_data_token[0]}/{sample_data_token[2]}/env/"
            )
            os.makedirs(vis_dir_seq, exist_ok=True)
            vis_dir_seq = os.path.join(
                vis_dir, f"{sample_data_token[0]}/{sample_data_token[2]}/withtraj/"
            )
            os.makedirs(vis_dir_seq, exist_ok=True)
            vis_dir_seq = os.path.join(
                vis_dir, f"{sample_data_token[0]}/{sample_data_token[2]}/all_cost/"
            )
            os.makedirs(vis_dir_seq, exist_ok=True)
            vis_dir_seq = os.path.join(
                vis_dir,
                f"{sample_data_token[0]}/{sample_data_token[2]}/rendered_freespace/",
            )
            os.makedirs(vis_dir_seq, exist_ok=True)
            vis_dir_seq = os.path.join(
                vis_dir,
                f"{sample_data_token[0]}/{sample_data_token[2]}/observed_sweep/",
            )
            os.makedirs(vis_dir_seq, exist_ok=True)
            vis_dir_seq = os.path.join(
                vis_dir, f"{sample_data_token[0]}/{sample_data_token[2]}/input_sweep/"
            )
            os.makedirs(vis_dir_seq, exist_ok=True)
            vis_dir_seq = os.path.join(
                vis_dir,
                f"{sample_data_token[0]}/{sample_data_token[2]}/inp_occ_render/",
            )
            os.makedirs(vis_dir_seq, exist_ok=True)
            vis_dir_seq = os.path.join(
                vis_dir,
                f"{sample_data_token[0]}/{sample_data_token[2]}/out_occ_render/",
            )
            os.makedirs(vis_dir_seq, exist_ok=True)
            # if sample_data_token[0] == '000104':
            #     print("Continuing ...")
            #     continue

            if args.compute_raydist_loss:
                lessptspath = f"/data3/tkhurana/datasets/once/data/{sample_data_token[0]}/lesspoints/{sample_data_token[1]}.npy"
                less_points = np.load(lessptspath).astype(np.float32)
                # print(less_points.shape)
                # less_points = less_points.reshape(output_origins[j].shape[0], -1, 3)
                # print("less points shape", less_points.shape)
                for tind in range(less_points.shape[0]):
                    if tind == 0:
                        new_less_points = np.concatenate((less_points[tind], np.full((less_points.shape[1], 1), tind)), axis=1)
                    else:
                        new_less_points = np.vstack([new_less_points, np.concatenate((less_points[tind], np.full((less_points.shape[1], 1), tind)), axis=1)])
                # print("after", new_less_points.shape)
                # print(new_less_points[:5])
                less_points = new_less_points[new_less_points[:, 2] != -1]
                # print("after after", less_points.shape)

            if plot_on:
                # tt = [2, 4, 6]
                tt = list(range(_n_output))
                for t in tt:
                    if il_costs is not None:
                        il_cost = il_costs[j, t]
                        zeros = np.zeros_like(il_cost)
                        il_cost = (
                            (il_cost - il_cost.min())
                            / (il_cost.max() - il_cost.min())
                            * 255
                        )
                        print("unique values in il_cosrt", np.unique(il_cost))
                        il_cost = cv2.applyColorMap(
                            il_cost.astype(np.uint8), cv2.COLORMAP_BONE
                        )
                        il_cost = cv2.cvtColor(il_cost, cv2.COLOR_BGR2GRAY)
                        cv2.imwrite(
                            f"{vis_dir}/{sample_data_token[0]}/{sample_data_token[2]}/il/1{t}.png",
                            rotate(il_cost[::-1], "residual"),
                        )
                        if t == 0:
                            first_il_cost = il_cost
                        if t < 5:
                            print(il_cost.dtype, whitebg.dtype)
                            il_cost_white = cv2.addWeighted(
                                first_il_cost, 0.2, whitebg, 0.8, 0
                            )
                            cv2.imwrite(
                                f"{vis_dir}/{sample_data_token[0]}/{sample_data_token[2]}/il/0{t}.png",
                                rotate(il_cost_white[::-1], "residual"),
                            )

                    if occ_costs is not None:
                        occ_cost = occ_costs[j, t]
                        c = 2
                        # occ_cost = 1 / (1 + (1/(1 /(1 + np.exp(- occ_cost))) - 1)**c)
                        ones = np.ones_like(occ_cost) * 255.0
                        zeros = np.zeros_like(occ_cost)
                        occ_cost = occ_cost / occ_cost.max() * 255.0
                        # occ_cost = np.clip(occ_cost, 50, 255)
                        occ_cost = 255.0 - occ_cost
                        occ_cost = np.dstack((occ_cost, occ_cost, ones))
                        # occ_cost = np.dstack([zeros, zeros, occ_cost])
                        # occ_cost = occ_cost[::-1] / occ_cost.max() * 255
                        cv2.imwrite(
                            f"{vis_dir}/{sample_data_token[0]}/{sample_data_token[2]}/occ/1{t}.png",
                            occ_cost[::-1],
                        )
                        if t == 0:
                            first_occ = occ_cost
                        if t < 5:
                            occ_cost = cv2.addWeighted(
                                first_occ.astype(np.uint8),
                                0.2,
                                np.dstack([whitebg, whitebg, whitebg]),
                                0.8,
                                0,
                            )
                            cv2.imwrite(
                                f"{vis_dir}/{sample_data_token[0]}/{sample_data_token[2]}/occ/0{t}.png",
                                rotate(occ_cost, "latent occupancy"),
                            )

                    if nvf_probs is not None:
                        nvf_prob = nvf_probs[j, t]
                        ones = np.where(nvf_prob.astype(np.uint8) == 0, 255.0, 255.0)
                        nvf_cost = nvf_prob * 255
                        nvf_prob = nvf_prob * 255.0
                        print("unique values in nvf_prob", np.unique(nvf_prob))
                        nvf_prob = np.dstack((nvf_prob, ones, nvf_prob))
                        cv2.imwrite(
                            f"{vis_dir}/{sample_data_token[0]}/{sample_data_token[2]}/nvf/1{t}.png",
                            nvf_prob[::-1],
                        )
                        print(f"{vis_dir}/{sample_data_token[0]}/{sample_data_token[2]}/nvf/1{t}.png")
                        nvf_cost = cv2.applyColorMap(
                            nvf_cost.astype(np.uint8), cv2.COLORMAP_BONE
                        )
                        nvf_cost = cv2.cvtColor(nvf_cost, cv2.COLOR_BGR2GRAY)
                        cv2.imwrite(
                            f"{vis_dir}/{sample_data_token[0]}/{sample_data_token[2]}/nvfcost/1{t}.png",
                            rotate(nvf_cost[::-1]),
                        )
                        if t < 5:
                            nvf_cost = cv2.addWeighted(
                                nvf_cost.astype(np.uint8), 0.3, whitebg, 0.7, 0
                            )
                            cv2.imwrite(
                                f"{vis_dir}/{sample_data_token[0]}/{sample_data_token[2]}/nvfcost/0{t}.png",
                                rotate(nvf_cost[::-1]),
                            )
                            nvf_prob = cv2.addWeighted(
                                nvf_prob.astype(np.uint8),
                                0.3,
                                np.dstack([whitebg, whitebg, whitebg]),
                                0.7,
                                0,
                            )
                            cv2.imwrite(
                                f"{vis_dir}/{sample_data_token[0]}/{sample_data_token[2]}/nvf/0{t}.png",
                                rotate(nvf_prob[::-1]),
                            )

                    if nvf_costs is not None:
                        nvf_cost = nvf_costs[j, t].astype(np.uint8)
                        print("unique values in nvf_cost", np.unique(nvf_cost))
                        nvf_cost = nvf_cost / nvf_cost.max()
                        ones = np.where(nvf_cost.astype(np.uint8) == 0, 255.0, 255.0)
                        nvf_cost = nvf_cost * 255.0
                        nvf_cost = np.dstack((nvf_cost, ones, nvf_cost))
                        cv2.imwrite(
                            f"{vis_dir}/{sample_data_token[0]}/{sample_data_token[2]}/env/1{t}.png",
                            rotate(nvf_cost[::-1]),
                        )
                        """
                        if t < 5:
                            nvf_cost = cv2.addWeighted(nvf_cost.astype(np.uint8), 0.2, whitebg, 0.8, 0)
                            cv2.imwrite(
                                f"{vis_dir}/{sample_data_token[0]}/{sample_data_token[2]}/env/0{t}.png",
                                rotate(nvf_cost[::-1]),
                            )
                        """

                    if obj_costs is not None:
                        obj_cost = obj_costs[j, t]
                        plt.imsave(
                            f"{vis_dir}/{sample_data_token[0]}/{sample_data_token[2]}/env/1{t}.png",
                            rotate(obj_cost[::-1]),
                        )
                        if t < 5:
                            obj_cost = cv2.addWeighted(obj_cost, 0.2, whitebg, 0.8, 0)
                            cv2.imwrite(
                                f"{vis_dir}/{sample_data_token[0]}/{sample_data_token[2]}/env/0{t}.png",
                                rotate(obj_cost[::-1]),
                            )

                    if costs is not None:
                        cost = costs[j, t]
                        cost = cost.max() - cost
                        cost = (cost - cost.min()) / (cost.max() - cost.min()) * 175
                        cost = 255.0 - cost
                        cost = cv2.applyColorMap(
                            cost.astype(np.uint8), cv2.COLORMAP_BONE
                        )
                        cost = cv2.cvtColor(cost, cv2.COLOR_BGR2GRAY)
                        output_plan = sampled_plans[j, best_plans[j, 0]]
                        ppx = (output_plan[: t + 1, 0] + 40.0) / 0.2
                        ppy = (output_plan[: t + 1, 1] + 70.4) / 0.2
                        print(output_plan.shape, ppx, ppy)
                        cost = np.dstack((cost, cost, cost))
                        for c in range(ppx.shape[0]):
                            cost = cv2.circle(
                                cost, (int(ppx[c]), int(ppy[c])), 4, (255, 0, 0), -1
                            )
                            if c == 0:
                                first_point = (int(ppx[c]), int(ppy[c]))
                            if c != 0:
                                second_point = (int(ppx[c]), int(ppy[c]))
                                cost = cv2.line(
                                    cost, first_point, second_point, (255, 0, 0), 2
                                )
                                first_point = second_point
                        # all_cost = il_cost / 2 + nvf_cost / 2
                        # all_cost = all_cost.astype(np.uint8)
                        cv2.imwrite(
                            f"{vis_dir}/{sample_data_token[0]}/{sample_data_token[2]}/withtraj/1{t}.png",
                            rotate(cost[::-1], "total cost + output trajectory"),
                        )
                        # cv2.imwrite(
                        #     f"{vis_dir}/{sample_data_token[0]}/{sample_data_token[2]}/all_cost/1{t}.png",
                        #     rotate(all_cost[::-1]),
                        # )
                        if t == 0:
                            first_cost = cost
                        if t < 5:
                            cost = cv2.addWeighted(
                                first_cost.astype(np.uint8),
                                0.2,
                                np.dstack([whitebg, whitebg, whitebg]),
                                0.8,
                                0,
                            )
                            cv2.imwrite(
                                f"{vis_dir}/{sample_data_token[0]}/{sample_data_token[2]}/withtraj/0{t}.png",
                                rotate(cost[::-1], "total cost + output trajectory"),
                            )
                            # all_cost = cv2.addWeighted(
                            #     all_cost.astype(np.uint8), 0.2, whitebg, 0.8, 0
                            # )
                            # cv2.imwrite(
                            #     f"{vis_dir}/{sample_data_token[0]}/{sample_data_token[2]}/all_cost/0{t}.png",
                            #     rotate(all_cost[::-1]),
                            # )

            if plot_on and render_freespace:
                # define some fixed quantities
                pc_range = [-40.0, -70.4, -2.0, 40.0, 70.4, 3.4]
                voxel_size = 0.2
                output_grid = [7, 704, 400]
                device = torch.device("cuda:0")

                offset = torch.nn.parameter.Parameter(
                    torch.Tensor(pc_range[:3])[None, None, :], requires_grad=False
                )
                scaler = torch.nn.parameter.Parameter(
                    torch.Tensor([voxel_size] * 3)[None, None, :], requires_grad=False
                )

                occ_prob = occ_probs[j].unsqueeze(0)
                output_origin = output_origins[j].unsqueeze(0)
                output_point = output_points[j].unsqueeze(0)
                input_point = input_points[j].unsqueeze(0)
                # output_points = generate_point_cloud(occ_prob[j]) # these will already be in the
                # BEV coords so no need to do the
                # next step for this
                output_point[:, :, :3] = (output_point[:, :, :3] - offset) / scaler
                input_point[:, :, :3] = (input_point[:, :, :3] - offset) / scaler
                output_origin[:, :, :3] = (output_origin[:, :, :3] - offset) / scaler

                # formula: probability = 1 - exp(-sigma)
                # shape: batch_size x height x width x time
                c = 2
                occ_prob_ = 1 / (1 + (1 / occ_prob - 1) ** c)
                sigma = -torch.log(1.0 - occ_prob_)
                sigma = sigma.to(device)
                output_origin = output_origin.to(device)
                output_point = output_point.to(device)
                input_point = input_point.to(device)
                print(output_origin.shape, output_point.shape, sigma.shape)

                # call rendering kernel
                pred_dist, gt_dist, loss, grad_sigma = renderer.render(
                    sigma, output_origin, output_point
                )

                # get things back onto CPU for plotting
                sigma = sigma.detach().cpu().numpy()
                occ_prob_ = occ_prob_.detach().cpu().numpy()
                origins = output_origin.detach().cpu().numpy()
                points = output_point.detach().cpu().numpy()
                inputpoints = input_point.detach().cpu().numpy()
                pred_dist = pred_dist.detach().cpu().numpy()
                num_points = points.shape[1]

                # plot rendered sweep
                pred_points = np.zeros((1, num_points, 2))

                # plot every lidar ray
                for n in range(1):
                    for t in range(7):
                        sigmat = sigma[0, t]
                        zeros = np.zeros_like(sigmat) + 255
                        print("unique values in sigmat", np.unique(sigmat))
                        cv2.imwrite(
                            f"{vis_dir}/{sample_data_token[0]}/{sample_data_token[2]}/occ/1{t}.png",
                            rotate(
                                np.dstack([255 - occ_prob_[0, t] * 255,
                                           255 - occ_prob_[0, t] * 255,
                                           zeros])[::-1],
                                "",
                            ),
                        )
                        '''cv2.imwrite(
                            f"{vis_dir}/{sample_data_token[0]}/{sample_data_token[2]}/occ/1{t}.png",
                            rotate(
                                np.dstack([zeros, zeros, occ_prob_[0, t] * 255])[::-1],
                                "",
                            ),
                        )
                        '''
                        # white bg: plt.imshow(np.dstack((zeros, 1 - occ_prob[0, t], 1 - occ_prob[0, t])))
                        # black bg: plt.imshow(np.dstack((occ_prob[0, t], zeros, zeros)))
                        idx = np.flatnonzero(points[n, :, 3] == t)
                        # print(points[n, idx, :2].shape, origins[n, t, None, :].shape)
                        unit_vector = points[n, idx, :2] - origins[n, t, None, :2]
                        unit_vector /= np.linalg.norm(
                            unit_vector, axis=-1, keepdims=True
                        )
                        pred_points[n, idx, :2] = (
                            origins[n, t, None, :2]
                            + unit_vector * pred_dist[n, idx, None]
                        )
                        ptsx = pred_points[n, idx, 0]
                        ptsy = pred_points[n, idx, 1]
                        indices = np.where(
                            (ptsx >= 0) & (ptsx < 400) & (ptsy >= 0) & (ptsy < 704)
                        )
                        ptsx = ptsx[indices].astype(int)
                        ptsy = ptsy[indices].astype(int)
                        zeros[ptsy, ptsx] = 255
                        cv2.imwrite(
                            f"{vis_dir}/{sample_data_token[0]}/{sample_data_token[2]}/rendered_freespace/1{t}.png",
                            rotate(zeros[::-1], "rendered sweep"),
                        )

                        # also plot the lidar sweep for this sample with ground points removed
                        X, Y, Z, _, labels = points[n, idx].T
                        print(np.unique(labels))
                        pt_map = np.zeros((704, 400)) + 255
                        Xi = X.astype(int)  # ((X + 40.0) / 0.2).astype(int)
                        Yi = Y.astype(int)  # ((Y + 70.4) / 0.2).astype(int)
                        mask = np.logical_and(
                            np.logical_and(0 <= Xi, Xi < 400),
                            np.logical_and(0 <= Yi, Yi < 704),
                        )
                        Yi1, Xi1 = Yi[mask], Xi[mask]
                        pt_map[Yi1, Xi1] = 0

                        # print(mask.shape, gseg.shape)
                        print("unique labels", np.unique(labels))
                        mask2 = np.logical_and(
                            mask, # np.logical_or(labels == 24, labels == 31)
                            np.logical_and(24 <= labels, labels <= 27))
                        Yi2, Xi2 = Yi[mask2], Xi[mask2]
                        pt_map[Yi2, Xi2] = 175
                        if t == 0:
                            background = np.dstack([pt_map, pt_map, pt_map])
                        img = np.dstack(
                            [np.zeros_like(zeros), zeros, occ_prob[0, t] * 255]
                        )
                        pt_map = np.dstack([pt_map, pt_map, pt_map])
                        img_moving = img + pt_map
                        img = img + background
                        img = np.clip(img, 0, 255)
                        img_moving = np.clip(img_moving, 0, 255)
                        img_moving = img_moving.astype(np.uint8)
                        img = img.astype(np.uint8)
                        print("img shape", img.shape)
                        # fillindices = np.any(background != 0, axis=-1)
                        # img[fillindices] = np.dstack([background[fillindices], background[fillindices], background[fillindices]])
                        cv2.imwrite(
                            f"{vis_dir}/{sample_data_token[0]}/{sample_data_token[2]}/observed_sweep/1{t}.png",
                            rotate(pt_map[::-1] , ""),
                        )
                        cv2.imwrite(
                            f"{vis_dir}/{sample_data_token[0]}/{sample_data_token[2]}/inp_occ_render/1{t}.png",
                            rotate(img[::-1], "latent occupancy + rendered sweep"),
                        )
                        cv2.imwrite(
                            f"{vis_dir}/{sample_data_token[0]}/{sample_data_token[2]}/out_occ_render/1{t}.png",
                            rotate(
                                img_moving[::-1], "latent occupancy + rendered sweep"
                            ),
                        )
                        if t == 0:
                            first_render = zeros
                            first_out = pt_map
                        if t < 5:
                            zeros = cv2.addWeighted(
                                first_render.astype(np.uint8), 0.2, whitebg, 0.8, 0
                            )
                            cv2.imwrite(
                                f"{vis_dir}/{sample_data_token[0]}/{sample_data_token[2]}/rendered_freespace/0{t}.png",
                                rotate(zeros[::-1], "rendered sweep"),
                            )
                            zeros = cv2.addWeighted(
                                first_out.astype(np.uint8),
                                0.2,
                                np.dstack([whitebg, whitebg, whitebg]),
                                0.8,
                                0,
                            )
                            cv2.imwrite(
                                f"{vis_dir}/{sample_data_token[0]}/{sample_data_token[2]}/observed_freespace/0{t}.png",
                                rotate(zeros[::-1], "groundtruth output sweep"),
                            )

                for n in range(1):
                    for t in range(12):
                        # also plot the lidar sweep for this sample with ground points removed
                        if t < 5:
                            timestamp = 4 - t
                        else:
                            timestamp = 0
                        idx = np.flatnonzero(inputpoints[n, :, 3] == timestamp)
                        X, Y, Z, _ = inputpoints[n, idx].T
                        print("unique points", np.unique(inputpoints[n, idx]))
                        pt_map = np.zeros((704, 400)) + 255
                        Xi = X.astype(int)  # ((X + 40.0) / 0.2).astype(int)
                        Yi = Y.astype(int)  # ((Y + 70.4) / 0.2).astype(int)
                        mask = np.logical_and(
                            np.logical_and(0 <= Xi, Xi < 400),
                            np.logical_and(0 <= Yi, Yi < 704),
                        )
                        Yi1, Xi1 = Yi[mask], Xi[mask]
                        pt_map[Yi1, Xi1] = 0

                        """
                        # print(mask.shape, gseg.shape)
                        print("unique labels", np.unique(labels))
                        mask2 = np.logical_and(mask, np.logical_or(
                            labels == 24, labels == 31)) # np.logical_and(24 <= labels, labels <= 27))
                        Yi2, Xi2 = Yi[mask2], Xi[mask2]
                        pt_map[Yi2, Xi2] = 12
                        """
                        pt_map = np.dstack([pt_map, pt_map, pt_map])
                        if t < 5:
                            cv2.imwrite(
                                f"{vis_dir}/{sample_data_token[0]}/{sample_data_token[2]}/input_sweep/0{t}.png",
                                rotate(pt_map[::-1], ""),
                            )
                        else:
                            pt_map = cv2.addWeighted(
                                pt_map.astype(np.uint8),
                                0.2,
                                np.dstack([whitebg, whitebg, whitebg]),
                                0.8,
                                0,
                            )
                            cv2.imwrite(
                                f"{vis_dir}/{sample_data_token[0]}/{sample_data_token[2]}/input_sweep/1{t-5}.png",
                                rotate(pt_map[::-1], "input sweep"),
                            )

            # rasterized collision ground truth
            obj_box_dir = f"{cfg['data_root']}/data/"
            obj_box_path = f"{obj_box_dir}/{sample_data_token[0]}/obj_boxes/{sample_data_token[1]}.bin"
            obj_boxes = np.fromfile(obj_box_path, dtype=bool).reshape((-1, 704, 400))

            # T tells us how many future frames we have expert data for
            T = len(obj_boxes)
            counts[:T] += 1
            unique = np.sort(obj_boxes.reshape((7, -1)), axis=1)
            unique = (unique[:, 1:] != unique[:, :-1]).sum(axis=1)
            coll_counts += unique

            # skip when gt plan is flawed (because of the limits of BEV planning)
            gt_plan = gt_plans[j]

            # compute L2 distance
            # best_plan = best_plans[j, 0]
            output_plan = sampled_plans[j, best_plans[j, 0]]
            if args.compute_dense_fvf_loss:
                nvf_prob = nvf_probs[j].reshape(7, -1)
                nvf_gt = nvf_gts[j].reshape(7, -1)
                nvf_pred = (nvf_prob > 0.5) + 0
                """
                for tind in range(7):
                    plt.imsave(f'/data3/tkhurana/gt_{tind}.png', nvf_gts[j][tind])
                    plt.imsave(f'/data3/tkhurana/prob_{tind}.png', nvf_probs[j][tind])
                    plt.imsave(f'/data3/tkhurana/pred_{tind}.png', (nvf_probs[j][tind] > 0.5) + 0)
                """

            if args.compute_raydist_loss:
                if 0:
                    print("cannot compute raydist loss without occ_probs")
                    exit(0)
                else:
                    pc_range = [-40.0, -70.4, -2.0, 40.0, 70.4, 3.4]
                    voxel_size = 0.2
                    output_grid = [7, 704, 400]
                    offset = torch.nn.parameter.Parameter(
                        torch.Tensor(pc_range[:3])[None, None, :], requires_grad=False
                    )
                    scaler = torch.nn.parameter.Parameter(
                        torch.Tensor([voxel_size] * 3)[None, None, :],
                        requires_grad=False,
                    )
                    less_points = torch.from_numpy(less_points).unsqueeze(0).type(dtype=torch.float32)
                    less_points[:, :, :3] = (less_points[:, :, :3] - offset) / scaler
                    output_origin = output_origins[j].unsqueeze(0)
                    output_origin[:, :, :3] = (
                        output_origin[:, :, :3] - offset
                    ) / scaler
                    # print(torch.log(1 - occ_probs[j]).dtype, output_origin.dtype, less_points.dtype)
                    c = 3
                    nvf_prob = torch.from_numpy(nvf_probs[j]).type(torch.float64)
                    nvf_prob = 1 / (1 + (1 / nvf_prob - 1) ** c)
                    # occ_prob = occ_probs[j].type(torch.float64)
                    # occ_prob = 1 / (1 + (1 / occ_prob - 1) ** c)
                    pred_dist, gt_dist, _, _ = renderer.render(
                        - torch.log(1 - nvf_prob).float().to(device).unsqueeze(0),
                        output_origin.float().to(device),
                        less_points.float().to(device),
                    )
                    # print("pred dist shape", pred_dist.shape)
                    pred_dist = pred_dist[0].detach().cpu().numpy()
                    gt_dist = gt_dist[0].detach().cpu().numpy()
                    indices = ~np.isnan(pred_dist)
                    pred_dist = pred_dist[indices]
                    gt_dist = gt_dist[indices]
                    # print("pred_dist", pred_dist[:10])
                    # print("gt_dist", gt_dist[:10])
                    less_points = less_points[:, indices, :]
                    # raydist_error += np.sum(np.abs(pred_dist - gt_dist), axis=(-1, -2))
                    # print(less_points[0][:10])
                    # print((less_points[..., 3].type(dtype=torch.int) == 2)[0])

                    pred_points = np.zeros((1, less_points.shape[1], 2))
                    gt_points = np.zeros((1, less_points.shape[1], 2))
                    for n in range(1):
                        for t in range(7):
                            sigmat = -np.log(1 - nvf_prob.numpy())[t]
                            zeros = np.zeros((sigmat.shape[0], sigmat.shape[1], 3)) + 255
                            # print("unique values in sigmat", np.unique(sigmat))
                            # cv2.imwrite(
                            #     f"{vis_dir}/{sample_data_token[0]}/{sample_data_token[2]}/occ/1{t}.png",
                            #     rotate(
                            #         np.dstack([zeros, zeros, occ_prob_[0, t] * 255])[::-1],
                            #         "",
                            #     ),
                            # )
                            # # white bg: plt.imshow(np.dstack((zeros, 1 - occ_prob[0, t], 1 - occ_prob[0, t])))
                            # black bg: plt.imshow(np.dstack((occ_prob[0, t], zeros, zeros)))
                            idx = np.flatnonzero(less_points[n, :, 3] == t)
                            # print(points[n, idx, :2].shape, origins[n, t, None, :].shape)
                            unit_vector = less_points[n, idx, :2] - output_origin[n, t, None, :2]
                            unit_vector /= np.linalg.norm(
                                unit_vector, axis=-1, keepdims=True
                            )
                            pred_points[n, idx, :2] = (
                                output_origin[n, t, None, :2]
                                + unit_vector * pred_dist[idx, None]
                            )
                            ptsx = pred_points[n, idx, 0]
                            ptsy = pred_points[n, idx, 1]
                            indices = np.where(
                                (ptsx >= 0) & (ptsx < 400) & (ptsy >= 0) & (ptsy < 704)
                            )
                            ptsx = ptsx[indices].astype(int)
                            ptsy = ptsy[indices].astype(int)
                            zeros[ptsy[:50], ptsx[:50], 0] = 0
                            zeros[ptsy[:50], ptsx[:50], 2] = 0
                            gt_points[n, idx, :2] = (
                                output_origin[n, t, None, :2]
                                + unit_vector * gt_dist[idx, None]
                            )
                            ptsx = gt_points[n, idx, 0]
                            ptsy = gt_points[n, idx, 1]
                            indices = np.where(
                                (ptsx >= 0) & (ptsx < 400) & (ptsy >= 0) & (ptsy < 704)
                            )
                            ptsx = ptsx[indices].astype(int)
                            ptsy = ptsy[indices].astype(int)
                            zeros[ptsy[:50], ptsx[:50], 1] = 0
                            zeros[ptsy[:50], ptsx[:50], 2] = 0
                            cv2.imwrite(
                                f"/data3/tkhurana/pred_gt_points_{t}.png", zeros,
                            )

                    for tind in range(7):
                        # print(tind, np.sum(pred_dist[(less_points[..., 3] == tind)[0]]), np.sum(gt_dist[(less_points[..., 3] == tind)[0]]), np.sum(np.abs(pred_dist[(less_points[..., 3] == tind)[0]] - gt_dist[(less_points[..., 3] == tind)[0]])), gt_dist[(less_points[..., 3] == tind)[0]].shape[0])
                        raydist_error[tind] += np.sum(np.abs(pred_dist[(less_points[..., 3] == tind)[0]] - gt_dist[(less_points[..., 3] == tind)[0]]) / gt_dist[(less_points[..., 3] == tind)[0]]) / gt_dist[(less_points[..., 3] == tind)[0]].shape[0]

            if plot_on:
                # double check the sampled trajectories by plotting them
                trajectory_path = f"{vis_dir}/{sample_data_token}_traj.jpg"
                print("Doing trajectory", trajectory_path)
                plt.figure(figsize=(20, 20))
                for sampled_plan in sampled_plans[j]:
                    plt.plot(sampled_plan[:, 0], sampled_plan[:, 1], c="g")
                plt.plot(output_plan[:, 0], output_plan[:, 1], linewidth=4, c="r")
                plt.plot(gt_plan[:, 0], gt_plan[:, 1], c="b")
                plt.grid(False)
                plt.axis("equal")
                plt.savefig(trajectory_path)
                plt.close()

            gt_box_coll = evaluate_box_coll(obj_boxes, gt_plan, _pc_range)
            if occ_probs is not None:
                obj_recall = evaluate_obj_recall(
                    obj_boxes, occ_probs[j]
                )  # this should be a 1 x 7 array

            # test ego-vehicle point against annotated object boxes
            ti = np.arange(T)
            yi = ((output_plan[:T, 1] - _pc_range[1]) / _voxel_size).astype(int)
            xi = ((output_plan[:T, 0] - _pc_range[0]) / _voxel_size).astype(int)
            # when the best plan is outside the boundary
            m1 = np.logical_and(
                np.logical_and(
                    _pc_range[1] <= output_plan[:T, 1],
                    output_plan[:T, 1] < _pc_range[4],
                ),
                np.logical_and(
                    _pc_range[0] <= output_plan[:T, 0],
                    output_plan[:T, 0] < _pc_range[3],
                ),
            )
            # exclude cases where even the expert trajectory collides (box)
            # obviously the expert did not crash
            # it only looks that way because we are considering bird's-eye view
            m1 = np.logical_and(m1, np.logical_not(gt_box_coll[ti]))
            # and the cases where the groundtruth trajectory is outside the boundary
            m1 = np.logical_and(
                np.logical_and(
                    m1,
                    np.logical_and(
                        _pc_range[1] <= gt_plan[:T, 1], gt_plan[:T, 1] < _pc_range[4]
                    ),
                ),
                np.logical_and(
                    _pc_range[0] <= gt_plan[:T, 0], gt_plan[:T, 0] < _pc_range[3]
                ),
            )
            obj_coll_sum[ti[m1]] += obj_boxes[ti[m1], yi[m1], xi[m1]].astype(int)

            if occ_probs is not None:
                # for evaluating the latent occupancy, we dont care if anythign lies outside BEV range
                obj_recall_sum += obj_recall

            # test ego-vehicle box against annotated object boxes
            # exclude cases where the expert trajectory collides (box)
            m2 = np.logical_not(gt_box_coll[ti])
            box_coll = evaluate_box_coll(obj_boxes, output_plan, _pc_range)
            obj_box_coll_sum[ti[m2]] += (box_coll[ti[m2]]).astype(int)
            l2_dist = np.sqrt(((output_plan[:, :2] - gt_plan[:, :2]) ** 2).sum(axis=-1))
            l2_dist_sum[ti[m1]] += l2_dist[ti[m1]]
            if args.compute_dense_fvf_loss:
                # print("min max values in nvf_prob", np.min(nvf_prob), np.max(nvf_prob))
                # print(
                #     "min max values in nvf_gt",
                #     np.unique(nvf_gt),
                #     np.min(nvf_gt),
                #     np.max(nvf_gt),
                # )
                # dense_fvf_loss[ti[m1]] += (
                #     np.sum(np.abs(nvf_prob - nvf_gt), axis=(1, 2)) / (704 * 400)
                # )[ti[m1]]
                # print(nvf_gt.shape, nvf_prob.shape)
                # print(sample_data_token)
                for tind in range(7):
                    # print(np.unique(nvf_gt[tind]), np.unique(nvf_prob[tind]), np.unique(nvf_pred[tind]))
                    dense_fvf_bce[tind] += sklearn.metrics.log_loss(nvf_gt[tind], nvf_prob[tind])
                    dense_fvf_f1[tind]  += sklearn.metrics.f1_score(nvf_gt[tind], nvf_pred[tind])
                    dense_fvf_ap[tind]  += sklearn.metrics.average_precision_score(nvf_gt[tind], nvf_prob[tind])

            if args.write_to_file:
                PC = obj_boxes[ti[m1], yi[m1], xi[m1]].astype(int)[-1]
                BC = (box_coll[ti[m2]]).astype(int)[-2]
                TC = unique[-2]
                L2 = l2_dist[ti[m1]][-2]
                dict_list.append(
                    {str(sample_data_token): [int(PC), int(BC), int(TC), float(L2)]}
                )

        print(
            f"{args.test_split} Epoch-{args.test_epoch},",
            f"Batch: {i+1}/{num_batch},",
            f"L2: {l2_dist_sum / counts},",
            f"Pt: {obj_coll_sum / coll_counts * 100},",
            f"Box: {obj_box_coll_sum / coll_counts * 100}",
            f"Rec: {obj_recall_sum / coll_counts}",
        )
        if args.compute_dense_fvf_loss:
            print(f"BCE: {dense_fvf_bce / counts},",
                  f"f1: {dense_fvf_f1 / counts},",
                  f"AP: {dense_fvf_ap / counts},",)
            dense_fvf_bce_ = dense_fvf_bce / counts
            dense_fvf_f1_  = dense_fvf_f1 / counts
            dense_fvf_ap_  = dense_fvf_ap / counts
            print(f"BCE: {np.sum(dense_fvf_bce_) / 7}",
                  f"F1 : {np.sum(dense_fvf_f1_) / 7}",
                  f"AP : {np.sum(dense_fvf_ap_) / 7}")
        if args.compute_raydist_loss:
            print(f"L1(RD): {raydist_error / counts},",)

    if args.write_to_file:
        with open(detailed_results_file, "w") as fout:
            json.dump(dict_list, fout)

    res_dir = f"{model_dir}/results"
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    if plot_on:
        return

    res_file = f"{res_dir}/{args.test_split}_epoch_all_metrics_recall_{args.suffix}_{args.test_epoch}.txt"
    with open(res_file, "w") as f:
        f.write(f"Split: {args.test_split}\n")
        f.write(f"Epoch: {args.test_epoch}\n")
        f.write(f"Counts: {counts}\n")
        f.write(f"Coll Counts: {coll_counts}\n")
        f.write(f"L2 distances: {l2_dist_sum / counts}\n")
        f.write(f"Point collision rates: {obj_coll_sum / coll_counts * 100}\n")
        f.write(f"Box collision rates: {obj_box_coll_sum / coll_counts * 100}\n")
        f.write(f"Point collisions: {obj_coll_sum}\n")
        f.write(f"Box collisions: {obj_box_coll_sum}\n")
        f.write(f"Object Recall: {obj_recall_sum / coll_counts}\n")
        if args.compute_dense_fvf_loss:
            f.write(f"BCE: {np.sum(dense_fvf_bce_) / 7}\n")
            f.write(f"F1: {np.sum(dense_fvf_f1_) / 7}\n")
            f.write(f"AP: {np.sum(dense_fvf_ap_) / 7}\n")
        if args.compute_raydist_loss:
            f.write(f"d-d^: {raydist_error / counts}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--test-split", type=str, required=True)
    parser.add_argument("--test-epoch", type=int, default=5)
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=36)
    parser.add_argument("--cache-on", action="store_true")
    parser.add_argument("--cache-every", type=int, default=1)
    parser.add_argument("--plot-on", action="store_true")
    parser.add_argument("--render-freespace", action="store_true")
    parser.add_argument("--write-to-file", action="store_true")
    parser.add_argument("--compute-dense-fvf-loss", action="store_true")
    parser.add_argument("--compute-raydist-loss", action="store_true")
    parser.add_argument("--dilate", action="store_true")
    parser.add_argument("--plot-every", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=18)
    parser.add_argument(
        "--sampled-trajectories",
        type=str,
        default="curves",
        choices=["curves", "data", "data+curves"],
    )
    parser.add_argument("--sample-set", type=str, default="")
    parser.add_argument("--suffix", type=str, default="")

    args = parser.parse_args()

    np.random.seed(0)
    torch.random.manual_seed(0)

    test(args)
