#!/usr/bin/env python3
import os
import re
import json
import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from nuscenes.nuscenes import NuScenes
from util.once_devkit.once import ONCE

from data import nuScenesDataset, ONCEDataset, CollateFn
from model import *


def make_data_loaders(args):
    dataset_kwargs = {
        "n_input": args.n_input,
        "n_samples": args.n_samples,
        "n_output": args.n_output,
        "train_on_all_sweeps": args.train_on_all_sweeps,
        "sampled_trajectories": args.sampled_trajectories,
        "sample_set": args.sample_set
    }
    data_loader_kwargs = {
        "pin_memory": False,  # NOTE
        "shuffle": True,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers
    }

    # select dataset
    if args.dataset == "nusc":
        nusc = NuScenes(args.data_version, args.data_root)
        train_dataset = nuScenesDataset(nusc, args.nusc_train_split, dataset_kwargs)
        val_dataset = nuScenesDataset(nusc, "val", dataset_kwargs)
    elif args.dataset == "once":
        once = ONCE(args.data_root)
        train_dataset = ONCEDataset(once, args.data_version, dataset_kwargs)
        val_dataset = ONCEDataset(once, "val", dataset_kwargs)
    else:
        print("Not Implemented!")
        exit(0)

    data_loaders = {
        # consider using all the frames only for training
        "train": DataLoader(train_dataset, collate_fn=CollateFn,
                            **data_loader_kwargs),
        "val": DataLoader(val_dataset, collate_fn=CollateFn,
                          **data_loader_kwargs)
    }
    return data_loaders


def mkdir_if_not_exists(d):
    if not os.path.exists(d):
        print(f"creating directory {d}")
        os.makedirs(d)


def resume_from_ckpts(ckpt_dir, model, optimizer, scheduler):
    if len(os.listdir(ckpt_dir)) > 0:
        pattern = re.compile(r"model_epoch_(\d+).pth")
        epochs = []
        for f in os.listdir(ckpt_dir):
            m = pattern.findall(f)
            if len(m) > 0:
                epochs.append(int(m[0]))
        resume_epoch = max(epochs)
        ckpt_path = f"{ckpt_dir}/model_epoch_{resume_epoch}.pth"
        print(f"Resume training from checkpoint {ckpt_path}")

        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        start_epoch = 1 + checkpoint['epoch']
        n_iter = checkpoint["n_iter"]
    else:
        start_epoch = 0
        n_iter = 0
    return start_epoch, n_iter


def train(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_count = torch.cuda.device_count()

    #
    data_loaders = make_data_loaders(args)

    # instantiate a model and a renderer
    _n_input, _n_output = args.n_input, args.n_output
    _pc_range, _voxel_size = args.pc_range, args.voxel_size
    if args.model_type == "vanilla":
        model = VanillaNeuralMotionPlanner(_n_input, _n_output, _pc_range, _voxel_size)
    elif args.model_type == "vf_guided":
        model = VFGuidedNeuralMotionPlanner(_n_input, _n_output, _pc_range, _voxel_size)
    elif args.model_type == "obj_guided":
        model = ObjGuidedNeuralMotionPlanner(_n_input, _n_output, _pc_range, _voxel_size)
    elif args.model_type == "vf_explicit":
        model = VFExplicitNeuralMotionPlanner(_n_input, _n_output, _pc_range, _voxel_size, args.nvf_loss_factor)
    elif args.model_type == "vf_supervised":
        model = VFSupervisedNeuralMotionPlanner(_n_input, _n_output, _pc_range, _voxel_size, args.nvf_loss_factor)
    elif args.model_type == "lat_occ_vf_supervised":
        model = LatOccVFSupervisedNeuralMotionPlanner(_n_input, _n_output, _pc_range, _voxel_size, args.nvf_loss_factor)
    else:
        raise NotImplementedError(f"{args.model_type} not implemented yet.")

    #
    model = model.to(device)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_start)

    # scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_epoch, gamma=args.lr_decay)

    # dump config
    mkdir_if_not_exists(args.model_dir)
    with open(f"{args.model_dir}/config.json", 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    # resume
    ckpt_dir = f"{args.model_dir}/ckpts"
    mkdir_if_not_exists(ckpt_dir)
    start_epoch, n_iter = resume_from_ckpts(ckpt_dir, model, optimizer, scheduler)

    # data parallel
    model = nn.DataParallel(model)

    #
    writer = SummaryWriter(f"{args.model_dir}/tf_logs")
    for epoch in range(start_epoch, args.num_epoch):
        for phase in ["train"]:
            data_loader = data_loaders[phase]
            if phase == "train":
                model.train()
            else:
                model.eval()

            sum_val_loss = {}
            num_batch = len(data_loader)
            num_example = len(data_loader.dataset)
            for i, batch in enumerate(data_loader):
                bs = len(batch["sample_data_tokens"])
                if bs < device_count:
                    print(f"Dropping the last batch of size {bs}")
                    continue

                # use the following to prevent overfitting
                if args.max_iters_per_epoch > 0 and i >= args.max_iters_per_epoch:
                    print(f"Breaking out because of exceeding {args.max_iters_per_epoch} iterations.")
                    break

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    results = model(batch, "train")
                    loss = results["loss"].mean()
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                print(f"Phase: {phase}, Iter: {n_iter},",
                      f"Epoch: {epoch}/{args.num_epoch},",
                      f"Batch: {i}/{num_batch}",
                      f"Loss: {loss.item():.6f}")

                if phase == "train":
                    n_iter += 1
                    for key in results:
                        writer.add_scalar(f"{phase}/{key}", results[key].mean().item(), n_iter)
                else:
                    for key in results:
                        if key not in sum_val_loss:
                            sum_val_loss[key] = 0.0
                        sum_val_loss[key] += (results[key].mean().item() * bs)

            if phase == "train":
                ckpt_path = f"{ckpt_dir}/model_epoch_{epoch}.pth"
                torch.save({
                    "epoch": epoch,
                    "n_iter": n_iter,
                    "model_state_dict": model.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                }, ckpt_path, _use_new_zipfile_serialization=False)
            else:
                for key in sum_val_loss:
                    mean_val_loss = sum_val_loss[key] / num_example
                    writer.add_scalar(f"{phase}/{key}", mean_val_loss, n_iter)

        scheduler.step()
    #
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    data_group = parser.add_argument_group("data")
    # data_group.add_argument("--dataset", type=str, default="once", choices=["nusc", "once"])
    # data_group.add_argument("--data-root", type=str, default="/data3/tkhurana/datasets/once")
    data_group.add_argument("--dataset", type=str, default="nusc", choices=["nusc", "once"])
    data_group.add_argument("--data-root", type=str, default="/data/nuScenes")
    data_group.add_argument("--nusc-train-split", type=str, default="train")
    data_group.add_argument("--data-version", type=str, choices=["v1.0-trainval", "v1.0-mini",
                            "labeled_train", "train_2k", "train_4k", "train_22k", "train_all", "train", "train_2k_unlabeled", "train_4k_unlabeled", "train_8k_unlabeled", "train_22k_unlabeled", "train_80k_unlabeled"])
    data_group.add_argument("--pc-range", type=float, nargs="+", default=[-40.0, -70.4, -2.0, 40.0, 70.4, 3.4])
    data_group.add_argument("--voxel-size", type=float, default=0.2)
    data_group.add_argument("--n-input", type=int, default=5)
    data_group.add_argument("--n-samples", type=int, default=1000)
    data_group.add_argument("--n-output", type=int, default=7)
    data_group.add_argument("--sampled-trajectories", type=str, choices=["curves", "data", "data+curves"], default="curves")
    data_group.add_argument("--sample-set", type=str, default="")

    model_group = parser.add_argument_group("model")
    model_group.add_argument("--model-type", type=str, required=True)
    model_group.add_argument("--train-on-all-sweeps", action="store_true")
    model_group.add_argument("--nvf-loss-factor", type=float, default=1.0)
    model_group.add_argument("--model-dir", type=str, required=True)
    model_group.add_argument("--optimizer", type=str, default="Adam")  # Adam with 5e-4
    model_group.add_argument("--lr-start", type=float, default=5e-4)
    model_group.add_argument("--lr-epoch", type=int, default=5)
    model_group.add_argument("--lr-decay", type=float, default=0.1)
    model_group.add_argument("--num-epoch", type=int, default=15)
    model_group.add_argument("--max-iters-per-epoch", type=int, default=-1)
    model_group.add_argument("--batch-size", type=int, default=36)
    model_group.add_argument("--num-workers", type=int, default=18)

    args = parser.parse_args()

    train(args)
