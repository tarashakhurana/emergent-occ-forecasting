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
# from model import OccupancyForecastingNetwork
# from model import NeuralMotionPlanner

from model import *

import matplotlib.pyplot as plt
import numpy as np

from skimage.draw import polygon
import sklearn
import cv2
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
        "train_on_all_sweeps": train_on_all_sweeps
    }
    data_loader_kwargs = {
        "pin_memory": False,  # NOTE
        "shuffle": True,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers
    }
    nusc = NuScenes(cfg["data_version"], cfg["data_root"])
    data_loader = DataLoader(nuScenesDataset(nusc, args.test_split, dataset_kwargs),
                             collate_fn=CollateFn, **data_loader_kwargs)
    return data_loader


def mkdir_if_not_exists(d):
    if not os.path.exists(d):
        print(f"creating directory {d}")
        os.makedirs(d)


def evaluate_box_coll(obj_boxes, trajectory, pc_range):
    xmin, ymin, _, xmax, ymax, _ = pc_range
    T, H, W = obj_boxes.shape
    collisions = np.full(T, False)
    for t in range(T):
        x, y, theta = trajectory[t]
        corners = np.array([
            (-0.8, -1.5, 1),  # back left corner
            (0.8, -1.5, 1),   # back right corner
            (0.8, 2.5, 1),    # front right corner
            (-0.8, 2.5, 1),   # front left corner
        ])
        tf = np.array([
            [np.cos(theta), -np.sin(theta), x],
            [np.sin(theta), np.cos(theta), y],
            [0, 0, 1],
        ])
        xx, yy = tf.dot(corners.T)[:2]

        yi = np.round((yy - ymin) / (ymax - ymin) * H).astype(int)
        xi = np.round((xx - xmin) / (xmax - xmin) * W).astype(int)
        rr, cc = polygon(yi, xi)
        I = np.logical_and(
            np.logical_and(rr >= 0, rr < H),
            np.logical_and(cc >= 0, cc < W),
        )
        collisions[t] = np.any(obj_boxes[t, rr[I], cc[I]])
    return collisions


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
    elif model_type == "lat_occ_vf_supervised_lat_occ":
        model = LatOccVFSupervisedLatOccNeuralMotionPlanner(
            _n_input, _n_output, _pc_range, _voxel_size, cfg["nvf_loss_factor"]
        )
    elif model_type == "lat_occ_flow_vf_supervised":
        model = LatOccFlowVFSupervisedNeuralMotionPlanner(_n_input, _n_output, _pc_range, _voxel_size, cfg["nvf_loss_factor"])
    elif model_type == "lat_occ_multiflow_vf_supervised":
        model = LatOccMultiFlowVFSupervisedNeuralMotionPlanner(_n_input, _n_output, _pc_range, _voxel_size, cfg["flow_mode"], cfg["nvf_loss_factor"])
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
    vis_dir = os.path.join(model_dir, "visuals", f"{args.test_split}_epoch_{args.test_epoch}")
    mkdir_if_not_exists(vis_dir)

    #
    counts = np.zeros(cfg["n_output"], dtype=int)
    l2_dist_sum = np.zeros(cfg["n_output"], dtype=float)
    obj_coll_sum = np.zeros(cfg["n_output"], dtype=int)
    obj_box_coll_sum = np.zeros(cfg["n_output"], dtype=int)
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
        cache_on = args.cache_on and (i % args.cache_every == 0)

        if plot_on and "il_cost" in results:
            il_costs = results["il_cost"].detach().cpu().numpy()
        else:
            il_costs = None

        if plot_on and "occ_cost" in results:
            occ_costs = results["occ_cost"].detach().cpu().numpy()
        else:
            occ_costs = None

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
            nvf_gts = np.where(nvf_gts == 1, 0, nvf_gts)
            nvf_gts += 1
            nvf_probs = 1 - nvf_probs
            nvf_gts = 1 - nvf_gts

        if args.compute_raydist_loss:
            nvf_probs = results["nvf_prob"].detach().cpu().numpy().astype("float64")

        for j, sample_data_token in enumerate(sample_data_tokens):
            # visualization:
            # - highlight the low cost regions (sub-zero)
            # - distinguish cost maps from different timestamps

            if args.compute_raydist_loss:
                lessptspath = f"/data3/tkhurana/datasets/nuScenes/lesspoints/v1.0-trainval/{sample_data_token}.npy"
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
                if il_costs is not None:
                    il_cost = np.concatenate(il_costs[j, tt], axis=-1)
                    plt.imsave(f"{vis_dir}/{sample_data_token}_il.png", il_cost[::-1])

                if occ_costs is not None:
                    occ_cost = np.concatenate(occ_costs[j, tt], axis=-1)
                    # plt.imsave(f"{vis_dir}/{sample_data_token}_occ.png", occ_cost[::-1])
                    plt.imsave(f"{vis_dir}/{sample_data_token}_env.png", occ_cost[::-1])

                if nvf_costs is not None:
                    nvf_cost = np.concatenate(nvf_costs[j, tt], axis=-1)
                    # plt.imsave(f"{vis_dir}/{sample_data_token}_nvf.png", nvf_cost[::-1])
                    plt.imsave(f"{vis_dir}/{sample_data_token}_env.png", nvf_cost[::-1])

                if obj_costs is not None:
                    obj_cost = np.concatenate(obj_costs[j, tt], axis=-1)
                    # plt.imsave(f"{vis_dir}/{sample_data_token}_obj.png", obj_cost[::-1])
                    plt.imsave(f"{vis_dir}/{sample_data_token}_env.png", obj_cost[::-1])

                if costs is not None:
                    cost = np.concatenate(costs[j, tt], axis=-1)
                    plt.imsave(f"{vis_dir}/{sample_data_token}.png", cost[::-1])

            # rasterized collision ground truth
            obj_box_path = f"{obj_box_dir}/{sample_data_token}.bin"
            obj_boxes = np.fromfile(obj_box_path, dtype=bool).reshape((-1, 704, 400))

            # T tells us how many future frames we have expert data for
            T = len(obj_boxes)
            # print("value of T is ", T)
            counts[:T] += 1

            # skip when gt plan is flawed (because of the limits of BEV planning)
            gt_plan = gt_plans[j]
            gt_box_coll = evaluate_box_coll(obj_boxes, gt_plan, _pc_range)

            # compute L2 distance
            # best_plan = best_plans[j, 0]
            output_plan = sampled_plans[j, best_plans[j, 0]]
            if args.compute_dense_fvf_loss:
                nvf_prob = nvf_probs[j].reshape(7, -1)
                nvf_gt = nvf_gts[j].reshape(7, -1)
                nvf_pred = (nvf_prob > 0.5) + 0
                print(sample_data_token, "unique nvf_gt", np.unique(nvf_gt))
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
                            # cv2.imwrite(
                            #     f"/data3/tkhurana/pred_gt_points_{t}.png", zeros,
                            # )

                    for tind in range(7):
                        # print(tind, np.sum(pred_dist[(less_points[..., 3] == tind)[0]]), np.sum(gt_dist[(less_points[..., 3] == tind)[0]]), np.sum(np.abs(pred_dist[(less_points[..., 3] == tind)[0]] - gt_dist[(less_points[..., 3] == tind)[0]])), gt_dist[(less_points[..., 3] == tind)[0]].shape[0])
                        raydist_error[tind] += np.sum(np.abs(pred_dist[(less_points[..., 3] == tind)[0]] - gt_dist[(less_points[..., 3] == tind)[0]]
                                                             ) / gt_dist[(less_points[..., 3] == tind)[0]]) / gt_dist[(less_points[..., 3] == tind)[0]].shape[0]

            l2_dist = np.sqrt(((output_plan[:, :2] - gt_plan[:, :2])**2).sum(axis=-1))
            l2_dist_sum[:T] += l2_dist[:T]

            # test ego-vehicle point against annotated object boxes
            ti = np.arange(T)
            yi = ((output_plan[:T, 1] - _pc_range[1]) / _voxel_size).astype(int)
            xi = ((output_plan[:T, 0] - _pc_range[0]) / _voxel_size).astype(int)
            # when the best plan is outside the boundary
            m1 = np.logical_and(
                np.logical_and(_pc_range[1] <= output_plan[:T, 1], output_plan[:T, 1] < _pc_range[4]),
                np.logical_and(_pc_range[0] <= output_plan[:T, 0], output_plan[:T, 0] < _pc_range[3])
            )
            # exclude cases where even the expert trajectory collides (box)
            # obviously the expert did not crash
            # it only looks that way because we are considering bird's-eye view
            m1 = np.logical_and(m1, np.logical_not(gt_box_coll[ti]))
            obj_coll_sum[ti[m1]] += obj_boxes[ti[m1], yi[m1], xi[m1]].astype(int)

            # test ego-vehicle box against annotated object boxes
            # exclude cases where the expert trajectory collides (box)
            m2 = np.logical_not(gt_box_coll[ti])
            box_coll = evaluate_box_coll(obj_boxes, output_plan, _pc_range)
            obj_box_coll_sum[ti[m2]] += (box_coll[ti[m2]]).astype(int)
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
                nvf_prob = nvf_prob.reshape((7, -1))
                print("unique values in nvf_gtf", np.unique(nvf_gt))
                for tind in range(7):
                    print(nvf_gt[tind].shape, nvf_prob[tind].shape, nvf_pred[tind].shape)
                    dense_fvf_bce[tind] += sklearn.metrics.log_loss(nvf_gt[tind], nvf_prob[tind])
                    dense_fvf_f1[tind] += sklearn.metrics.f1_score(nvf_gt[tind], nvf_pred[tind])
                    dense_fvf_ap[tind] += sklearn.metrics.average_precision_score(nvf_gt[tind], nvf_prob[tind])

        print(f"{args.test_split} Epoch-{args.test_epoch},",
              f"Batch: {i+1}/{num_batch},",
              f"L2: {l2_dist_sum / counts},",
              f"Pt: {obj_coll_sum / counts * 100},",
              f"Box: {obj_box_coll_sum / counts * 100}")
        if args.compute_dense_fvf_loss:
            print(f"BCE: {dense_fvf_bce / counts},",
                  f"f1: {dense_fvf_f1 / counts},",
                  f"AP: {dense_fvf_ap / counts},",)
            dense_fvf_bce_ = dense_fvf_bce / counts
            dense_fvf_f1_ = dense_fvf_f1 / counts
            dense_fvf_ap_ = dense_fvf_ap / counts
            print(f"BCE: {np.sum(dense_fvf_bce_) / 7}",
                  f"F1 : {np.sum(dense_fvf_f1_) / 7}",
                  f"AP : {np.sum(dense_fvf_ap_) / 7}")
        if args.compute_raydist_loss:
            print(f"L1(RD): {raydist_error / counts},",)

    res_dir = f"{model_dir}/results"
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    res_file = f"{res_dir}/{args.test_split}_epoch_{args.test_epoch}.txt"
    with open(res_file, "w") as f:
        f.write(f"Split: {args.test_split}\n")
        f.write(f"Epoch: {args.test_epoch}\n")
        f.write(f"Counts: {counts}\n")
        f.write(f"L2 distances: {l2_dist_sum / counts}\n")
        f.write(f"Point collision rates: {obj_coll_sum / counts * 100}\n")
        f.write(f"Box collision rates: {obj_box_coll_sum / counts * 100}\n")
        if args.compute_dense_fvf_loss:
            f.write(f"BCE: {dense_fvf_bce / counts}\n")
            f.write(f"F1: {dense_fvf_f1 / counts}\n")
            f.write(f"AP: {dense_fvf_ap / counts}\n")
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
    parser.add_argument("--plot-every", type=int, default=1)
    parser.add_argument("--compute-dense-fvf-loss", action="store_true")
    parser.add_argument("--compute-raydist-loss", action="store_true")
    parser.add_argument("--num-workers", type=int, default=18)

    args = parser.parse_args()

    np.random.seed(0)
    torch.random.manual_seed(0)

    test(args)
