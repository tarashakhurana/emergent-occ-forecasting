"""
  Data loader
"""
import os
import ipdb
import time
import pickle
import torch
import struct
import plyfile
import json
import numpy as np
import warnings
from pyquaternion import Quaternion
from torch.utils.data import Dataset
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.eval.common.utils import quaternion_yaw
import matplotlib.pyplot as plt

import sampler as trajectory_sampler


def locate_message(utimes, utime):
    i = np.searchsorted(utimes, utime)
    if i > 0 and utime - utimes[i - 1] < utimes[i] - utime:
        i -= 1
    return i


def find_nearest(array, value):
    array = np.asarray(array, dtype=int)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


class MyLidarPointCloud(LidarPointCloud):
    def remove_ego(self):
        ego_mask = np.logical_and(
            np.logical_and(-0.8 <= self.points[0], self.points[0] <= 0.8),
            np.logical_and(-1.5 <= self.points[1], self.points[1] <= 2.5),
        )
        self.points = self.points[:, np.logical_not(ego_mask)]


def CollateFn(batch):
    examples = {
        "scene_tokens": [],
        "sample_data_tokens": [],
        "input_points": [],
        "input_origin": [],
        "sampled_trajectories": [],
        "drive_commands": [],
        "output_origins": [],
        "output_points": [],
        "gt_trajectories": [],
        "obj_boxes": [],
        "obj_shadows": [],
        "fvf_maps": [],
    }

    max_n_input_points = max([len(example["input_points"]) for example in batch])
    max_n_output_points = max([len(example["output_points"]) for example in batch])

    examples = {
        "scene_tokens": [example["scene_token"] for example in batch],
        "sample_data_tokens": [example["sample_data_token"] for example in batch],
        "input_points": torch.stack(
            [
                torch.nn.functional.pad(
                    example["input_points"],
                    (0, 0, 0, max_n_input_points - len(example["input_points"])),
                    mode="constant",
                    value=-1,
                )
                for example in batch
            ]
        ),
        # "input_origins": torch.stack([example["input_origin"] for example in batch]),
        "sampled_trajectories_fine": torch.stack(
            [example["sampled_trajectories_fine"] for example in batch]
        ),
        "sampled_trajectories": torch.stack(
            [example["sampled_trajectories"] for example in batch]
        ),
        "drive_commands": torch.stack([example["drive_command"] for example in batch]),
        "output_origins": torch.stack([example["output_origin"] for example in batch]),
        "output_points": torch.stack(
            [
                torch.nn.functional.pad(
                    example["output_points"],
                    (0, 0, 0, max_n_output_points - len(example["output_points"])),
                    mode="constant",
                    value=-1,
                )
                for example in batch
            ]
        ),
        "gt_trajectories": torch.stack([example["gt_trajectory"] for example in batch]),
        "obj_boxes": torch.stack([example["obj_boxes"] for example in batch]),
        "obj_shadows": torch.stack([example["obj_shadows"] for example in batch]),
        "fvf_maps": torch.stack([example["fvf_maps"] for example in batch]),
    }

    # print("shape inside collate fn", examples["sampled_trajectories"].shape)

    examples["scene_tokens"] = [example["scene_token"] for example in batch]

    return examples


class MyLidarPointCloud(LidarPointCloud):
    def remove_ego(self):
        ego_mask = np.logical_and(
            np.logical_and(-0.8 <= self.points[0], self.points[0] <= 0.8),
            np.logical_and(-1.5 <= self.points[1], self.points[1] <= 2.5),
        )
        self.points = self.points[:, np.logical_not(ego_mask)]


class ONCEDataset(Dataset):
    N_SWEEPS_PER_SAMPLE = 1
    SAMPLE_INTERVAL = 0.5  # second

    def __init__(self, once, once_split, kwargs, seed=0):
        super(ONCEDataset, self).__init__()

        # set seed for split
        np.random.seed(seed)

        self.once = once
        self.data_root = self.once.data_root  # check if dataset_root or data_root
        self.data_split = once_split

        # number of input samples
        self.n_input = kwargs["n_input"]

        # number of sampled trajectories
        self.n_samples = kwargs["n_samples"]

        # type of sampled trajectories
        self.sample_type = kwargs["sampled_trajectories"]

        # the data-driven trajectory set
        self.sample_set = kwargs["sample_set"]

        # number of output samples
        self.n_output = kwargs["n_output"]
        assert self.n_output == 7

        #
        self.train_on_all_sweeps = kwargs["train_on_all_sweeps"]

        # TODO
        blacklist = ["000104"]

        # NOTE: use the official split (minus the ones in the blacklist)
        split_info = getattr(self.once, f"{self.data_split}_info")
        if "scene_token" in kwargs and kwargs["scene_token"] != "":
            scene_name = kwargs["scene_token"]
            scene = split_info[scene_name]
            scene["seq_id"] = scene_name
            scenes = [scene]
        else:
            scene_names = split_info.keys()
            scenes = []
            for scene_name in scene_names:
                scene = split_info[scene_name]
                scene["seq_id"] = scene_name
                if scene_name not in blacklist:
                    scenes.append(scene)

        # list all sample data
        # make a list of every scene's identifier + every scene's frames' LIDAR filenames/file contents?
        self.valid_index = []
        self.flip_flags = []
        self.scene_tokens = []
        self.sample_data_tokens = []
        for i, scene in enumerate(scenes):
            scene_token = scene["seq_id"]
            frame_list = scene["frame_list"]
            if len(frame_list) == 0:
                continue
            j = 0
            first_sample = scene[frame_list[j]]
            sample_data_token = [scene_token, frame_list[j], j]
            # flip x-y axis for once but set to False for now
            flip_flag = False
            # record the token of every key frame
            start_index = len(self.sample_data_tokens)
            while j < len(frame_list):
                sample_data_token = [scene_token, frame_list[j], j]
                sample_data = scene[frame_list[j]]
                if (
                    True
                ):  # (self.data_split == "train" and self.train_on_all_sweeps): # or (sample_data["is_key_frame"]):
                    self.flip_flags.append(flip_flag)
                    self.scene_tokens.append(scene_token)
                    self.sample_data_tokens.append(sample_data_token)
                j += 1
            end_index = len(self.sample_data_tokens)

            # use timestamps to collect the valid indices
            for frameid in range(
                start_index + self.n_input - 1, end_index - self.n_output + 1
            ):
                start_timestamp = self.sample_data_tokens[frameid - self.n_input + 1][1]
                end_timestamp = self.sample_data_tokens[frameid + self.n_output - 1][1]
                condition = int(end_timestamp) - int(start_timestamp) < 5010
                if condition:
                    self.valid_index.append(frameid)

            # NOTE: make sure we have enough number of sweeps for input and output
            # if self.data_split == "train" and self.train_on_all_sweeps:
            #     valid_start_index = start_index + self.n_input - 1
            #     valid_end_index = end_index - (self.n_output - 1) * self.N_SWEEPS_PER_SAMPLE
            # else:
            #     n_input_samples = self.n_input // self.N_SWEEPS_PER_SAMPLE
            #     valid_start_index = start_index + n_input_samples
            #     valid_end_index = end_index - self.n_output + 1
            # self.valid_index += list(range(valid_start_index, valid_end_index))

        self._n_examples = len(self.valid_index)

        # collect the set of train trajectory samples
        print(
            f"{self.data_root}/train_trajectories_by_long_vel_ang_{self.sample_set}.json"
        )
        if os.path.exists(
            f"{self.data_root}/train_trajectories_by_long_vel_ang_{self.sample_set}.json"
        ):
            self.train_trajectories = json.load(
                open(
                    f"{self.data_root}/train_trajectories_by_long_vel_ang_{self.sample_set}.json",
                    "rb",
                )
            )
            superkeys_to_delete = []
            for key in self.train_trajectories:
                value = self.train_trajectories[key]
                keys_to_delete = []
                for key2 in value:
                    if len(value[key2]) < 200:
                        # print(f"{len(value[key2])} train trajectories for {key} velocity and {key2} angle")
                        keys_to_delete.append(key2)
                for keytd in keys_to_delete:
                    _ = value.pop(keytd)
                    if len(value.keys()) == 0:
                        superkeys_to_delete.append(key)
            for keytd in superkeys_to_delete:
                _ = self.train_trajectories.pop(keytd)

        print(
            f"{self.data_split}: {self._n_examples} valid samples over {len(scenes)} scenes"
        )

    def __len__(self):
        return self._n_examples

    def get_global_pose(self, sd_token, inverse=False):
        split_info = getattr(self.once, f"{self.data_split}_info")
        sd_pose = split_info[sd_token[0]][sd_token[1]]["pose"]
        scene_names = split_info.keys()
        if inverse is False:
            global_from_curr = transform_matrix(
                sd_pose[-3:], Quaternion(sd_pose[:4]), inverse=False
            )
            pose = global_from_curr
        else:
            curr_from_global = transform_matrix(
                sd_pose[-3:], Quaternion(sd_pose[:4]), inverse=True
            )
            pose = curr_from_global
        return pose

    def load_ground_segmentation(self, sample_data_token):
        path = f"{self.once.data_root}/{sample_data_token[0]}/grndseg/{sample_data_token[1]}.bin"
        labels = np.fromfile(path, np.uint8)
        return labels

    def write_pointcloud(self, filename, xyz_points, rgb_points=None):
        """ creates a .pkl file of the point clouds generated

        """

        assert xyz_points.shape[1] == 3, "Input XYZ points should be Nx3 float array"
        if rgb_points is None:
            rgb_points = np.ones(xyz_points.shape).astype(np.uint8) * 255
        assert (
            xyz_points.shape == rgb_points.shape
        ), "Input RGB colors should be Nx3 float array and have same size as input XYZ points"

        # Write header of .ply file
        fid = open(filename, "wb")
        fid.write(bytes("ply\n", "utf-8"))
        fid.write(bytes("format binary_little_endian 1.0\n", "utf-8"))
        fid.write(bytes("element vertex %d\n" % xyz_points.shape[0], "utf-8"))
        fid.write(bytes("property float x\n", "utf-8"))
        fid.write(bytes("property float y\n", "utf-8"))
        fid.write(bytes("property float z\n", "utf-8"))
        fid.write(bytes("property uchar red\n", "utf-8"))
        fid.write(bytes("property uchar green\n", "utf-8"))
        fid.write(bytes("property uchar blue\n", "utf-8"))
        fid.write(bytes("end_header\n", "utf-8"))

        # Write 3D points to .ply file
        for i in range(xyz_points.shape[0]):
            fid.write(
                bytearray(
                    struct.pack(
                        "fffccc",
                        xyz_points[i, 0],
                        xyz_points[i, 1],
                        xyz_points[i, 2],
                        rgb_points[i, 0].tostring(),
                        rgb_points[i, 1].tostring(),
                        rgb_points[i, 2].tostring(),
                    )
                )
            )
        fid.close()

    def load_future_visible_freespace(self, sample_data_token):
        path = f"{self.once.data_root}/{sample_data_token[0]}/fvfmaps/{sample_data_token[1]}.bin"
        if os.path.exists(path):
            fvf_maps = np.fromfile(path, dtype=np.int8)
            fvf_maps = fvf_maps.reshape((7, 704, 400))
        else:
            fvf_maps = np.zeros((7, 704, 400), dtype=np.int8)
            warnings.warn(f"Cannot find fvf_maps at {path}")
        return fvf_maps

    def load_object_boxes(self, sample_data_token):
        path = f"{self.once.data_root}/{sample_data_token[0]}/obj_boxes/{sample_data_token[1]}.bin"
        if os.path.exists(path):
            obj_boxes = np.fromfile(path, dtype=bool)
            obj_boxes = obj_boxes.reshape((7, 704, 400))
        else:
            obj_boxes = np.zeros((7, 704, 400))
        return obj_boxes

    def load_object_shadows(self, sample_data_token):
        path = f"{self.once.data_root}/{sample_data_token[0]}/obj_shadows/{sample_data_token[1]}.bin"
        if os.path.exists(path):
            obj_shadows = np.fromfile(path, dtype=bool)
            obj_shadows = obj_shadows.reshape((7, 704, 400))
        else:
            obj_shadows = np.zeros((7, 704, 400))
        return obj_shadows

    def transform_once_point_cloud(self, curr_lidar_pc, ref_from_curr):
        curr_lidar_pc = curr_lidar_pc.T
        curr_lidar_pc[:3, :] = ref_from_curr.dot(
            np.vstack((curr_lidar_pc[:3, :], np.ones(curr_lidar_pc.shape[1])))
        )[:3, :]
        curr_lidar_pc = curr_lidar_pc.T
        return curr_lidar_pc

    def remove_ego_once(self, curr_lidar_pc):
        ego_mask = np.logical_and(
            np.logical_and(-0.8 <= curr_lidar_pc[:, 0], curr_lidar_pc[:, 0] <= 0.8),
            np.logical_and(-2.5 <= curr_lidar_pc[:, 1], curr_lidar_pc[:, 1] <= 2.5),
        )
        curr_lidar_pc = curr_lidar_pc[np.logical_not(ego_mask)]
        return curr_lidar_pc

    def __getitem__(self, idx):
        ref_index = self.valid_index[idx]

        ref_sd_token = self.sample_data_tokens[ref_index]
        ref_scene_token = self.scene_tokens[ref_index]
        flip_flag = self.flip_flags[ref_index]

        # reference coordinate frame
        ref_from_global = self.get_global_pose(ref_sd_token, inverse=True)

        # NOTE: input
        input_sds = []
        input_sd_tokens = []
        sd_token = ref_sd_token
        split_info = getattr(self.once, f"{self.data_split}_info")
        while len(input_sds) < self.n_input:
            curr_sd = split_info[sd_token[0]][sd_token[1]]
            input_sds.append(curr_sd)
            input_sd_tokens.append(sd_token)
            sd_token = [
                sd_token[0],
                split_info[sd_token[0]]["frame_list"][sd_token[2] - 1],
                sd_token[2] - 1,
            ]
            assert sd_token[2] >= -1

        # call out when we have less than the desired number of input sweeps
        # if len(input_sds) < self.n_input:
        #     warnings.warn(f"The number of input sweeps {len(input_sds)} is less than {self.n_input}.", RuntimeWarning)

        # get input sweep frames
        input_points_list = []
        input_origin_list = []
        for i, curr_sd in enumerate(input_sds):
            # load the current lidar sweep
            curr_sd_token = input_sd_tokens[i]
            curr_lidar_pc = self.once.load_point_cloud(
                curr_sd_token[0], curr_sd_token[1]
            )

            # remove ego returns
            curr_lidar_pc = self.remove_ego_once(curr_lidar_pc)

            # transform from the current lidar frame to global and then to the reference lidar frame
            global_from_curr = self.get_global_pose(curr_sd_token, inverse=False)
            ref_from_curr = ref_from_global.dot(global_from_curr)
            curr_lidar_pc = self.transform_once_point_cloud(
                curr_lidar_pc, ref_from_curr
            )

            # NOTE: flip the x and y axes
            curr_lidar_pc[:, 0] *= -1
            curr_lidar_pc[:, 1] *= -1
            ref_from_curr[0, 3] *= -1
            ref_from_curr[1, 3] *= -1

            #
            origin = np.array(ref_from_curr[:3, 3])
            points = np.asarray(curr_lidar_pc[:, :3])
            tindex = np.full((len(points), 1), i)
            points = np.concatenate((points, tindex), axis=1)

            #
            input_points_list.append(points.astype(np.float32))
            input_origin_list.append(origin.astype(np.float32))

        # NOTE: output
        # get output sample frames and ground truth trajectory
        output_origin_list = []
        output_points_list = []
        gt_trajectory = np.zeros((self.n_output, 3), np.float64)
        for i in range(self.n_output):
            if self.data_split == "train" and self.train_on_all_sweeps:
                index = ref_index + i * self.N_SWEEPS_PER_SAMPLE
            else:
                index = ref_index + i

            # if this exists a valid target
            if (
                index < len(self.scene_tokens)
                and self.scene_tokens[index] == ref_scene_token
            ):
                curr_sd_token = self.sample_data_tokens[index]
                curr_sd = split_info[curr_sd_token[0]][curr_sd_token[1]]
                # load the current lidar sweep
                curr_lidar_pc = self.once.load_point_cloud(
                    curr_sd_token[0], curr_sd_token[1]
                )

                # remove ego returns
                # curr_lidar_pc = self.remove_ego_once(curr_lidar_pc)

                # transform from the current lidar frame to global and then to the reference lidar frame
                global_from_curr = self.get_global_pose(curr_sd_token, inverse=False)
                ref_from_curr = ref_from_global.dot(global_from_curr)
                curr_lidar_pc = self.transform_once_point_cloud(
                    curr_lidar_pc, ref_from_curr
                )

                #
                theta = quaternion_yaw(Quaternion(matrix=ref_from_curr))

                # NOTE: flip the x and y axes to match the nuScenes coordinates
                ref_from_curr[0, 3] *= -1
                ref_from_curr[1, 3] *= -1
                curr_lidar_pc[:, 0] *= -1
                curr_lidar_pc[:, 1] *= -1
                theta += np.pi

                origin = np.array(ref_from_curr[:3, 3])

                points = np.array(curr_lidar_pc[:, :3])
                gt_trajectory[i, :] = [origin[0], origin[1], theta]

                tindex = np.full(len(points), i)

                labels = self.load_ground_segmentation(curr_sd_token)
                assert len(labels) == len(points)
                mask = np.logical_and(labels >= 1, labels <= 31)

                points = np.concatenate(
                    (points, tindex[:, None], labels[:, None]), axis=1
                )
                points = points[mask, :]

            else:  # filler
                raise RuntimeError(f"The {i}-th output frame is not available")
                origin = np.array([0.0, 0.0, 0.0])
                points = np.full((0, 5), -1)

            # origin
            output_origin_list.append(origin.astype(np.float32))

            # points
            output_points_list.append(points.astype(np.float32))

        """
        if ref_sd_token[1] == '1616617840000': # plot point clouds to verify
            ply_dir =  f'./videos_new/once/{ref_scene_token}/logs_ply/'
            os.makedirs(ply_dir, exist_ok=True)
            inputplyfile = f'./videos_new/once/{ref_scene_token}/logs_ply/{ref_sd_token[1]}_input.ply'
            outputplyfile = f'./videos_new/once/{ref_scene_token}/logs_ply/{ref_sd_token[1]}_output.ply'

            print(len(input_points_list), input_points_list[0].shape, inputplyfile)
            print(len(output_points_list), output_points_list[0].shape, outputplyfile)

            input_ply = np.vstack(input_points_list)[:, :3]
            output_ply = np.vstack(output_points_list)[:, :3]

            self.write_pointcloud(inputplyfile, input_ply)
            self.write_pointcloud(outputplyfile, output_ply)
        """
        """
        input_ply = np.array(input_ply, dtype=[("x", np.dtype("float32")),
                                               ("y", np.dtype("float32")),
                                               ("z", np.dtype("float32"))])
        output_ply = np.array(output_ply, dtype=[("x", np.dtype("float32")),
                                               ("y", np.dtype("float32")),
                                               ("z", np.dtype("float32"))])

        elements = plyfile.PlyElement.describe(input_ply, "vertex")
        plyfile.PlyData([elements]).write(inputplyfile)
        elements = plyfile.PlyElement.describe(output_ply, "vertex")
        plyfile.PlyData([elements]).write(outputplyfile)

        """
        # NOTE: trajectory sampling

        # initial speed
        v = (input_origin_list[0] - input_origin_list[1]) / 0.5
        v0 = v[1]  # [1] means longitudinal velocity (velocity along y)

        # sample from a set of curves
        if self.sample_type == "curves":
            # curvature (positive: turn left)
            steering = 0.0  # np.random.normal(0, 6)
            if flip_flag:
                steering *= -1
            Kappa = 2 * steering / 2.588

            # initial state
            T0 = np.array([0.0, 1.0])  # define front
            N0 = (
                np.array([1.0, 0.0]) if Kappa <= 0 else np.array([-1.0, 0.0])
            )  # define side

            #
            t_start = 0  # second
            t_end = (self.n_output - 1) * self.SAMPLE_INTERVAL  # second
            t_interval = self.SAMPLE_INTERVAL / 10
            tt = np.arange(t_start, t_end + t_interval, t_interval)
            sampled_trajectories_fine = trajectory_sampler.sample(
                v0, Kappa, T0, N0, tt, self.n_samples
            )
            sampled_trajectories = sampled_trajectories_fine[:, ::10]

        # sample from training data
        elif self.sample_type == "data":
            ang = (
                np.arctan2(
                    input_origin_list[0][1] - input_origin_list[1][1],
                    input_origin_list[0][0] - input_origin_list[1][0],
                )
                + np.pi
            )
            v0 = find_nearest(list(self.train_trajectories.keys()), v0)
            v0 = int(np.ceil(v0 / 2) * 2)
            ang = find_nearest(list(self.train_trajectories[str(v0)].keys()), ang)
            sampled_trajectories_fine = np.array(
                self.train_trajectories[str(v0)][str(int(ang))]
            )
            if "train" in self.data_split:
                intermediate_diff = np.abs(
                    sampled_trajectories_fine - gt_trajectory[None, ...]
                )
                intermediate = np.sum(intermediate_diff[:, :, :2], axis=-1)
                intermediate = np.sum(intermediate, axis=-1)
                index = intermediate.argmin()
                sampled_trajectories_fine = np.delete(
                    sampled_trajectories_fine, index, axis=0
                )
            indices = np.random.choice(
                sampled_trajectories_fine.shape[0], size=200, replace=False
            )
            sampled_trajectories_fine = sampled_trajectories_fine[indices]
            sampled_trajectories = sampled_trajectories_fine.copy()

        elif self.sample_type == "data+curves":
            # curvature (positive: turn left)
            steering = 0.0  # np.random.normal(0, 6)
            if flip_flag:
                steering *= -1
            Kappa = 2 * steering / 2.588

            # initial state
            T0 = np.array([0.0, 1.0])  # define front
            N0 = (
                np.array([1.0, 0.0]) if Kappa <= 0 else np.array([-1.0, 0.0])
            )  # define side

            #
            t_start = 0  # second
            t_end = (self.n_output - 1) * self.SAMPLE_INTERVAL  # second
            t_interval = self.SAMPLE_INTERVAL / 10
            tt = np.arange(t_start, t_end + t_interval, t_interval)
            sampled_trajectories_fine = trajectory_sampler.sample(
                v0, Kappa, T0, N0, tt, self.n_samples
            )
            sampled_trajectories = sampled_trajectories_fine[:, ::10]

            ang = (
                np.arctan2(
                    input_origin_list[0][1] - input_origin_list[1][1],
                    input_origin_list[0][0] - input_origin_list[1][0],
                )
                + np.pi
            )
            v0 = find_nearest(list(self.train_trajectories.keys()), v0)
            v0 = int(np.ceil(v0 / 2) * 2)
            ang = find_nearest(list(self.train_trajectories[str(v0)].keys()), ang)
            sampled_trajectories_data = np.array(
                self.train_trajectories[str(v0)][str(int(ang))]
            )
            if "train" in self.data_split:
                intermediate_diff = np.abs(
                    sampled_trajectories_data - gt_trajectory[None, ...]
                )
                intermediate = np.sum(intermediate_diff[:, :, :2], axis=-1)
                intermediate = np.sum(intermediate, axis=-1)
                index = intermediate.argmin()
                sampled_trajectories_data = np.delete(
                    sampled_trajectories_data, index, axis=0
                )
            indices = np.random.choice(
                sampled_trajectories_data.shape[0], size=200, replace=False
            )
            sampled_trajectories_data = sampled_trajectories_data[indices]
            # print(sampled_trajectories.shape, sampled_trajectories_data.shape, sampled_trajectories_fine.shape)
            sampled_trajectories = np.vstack(
                (sampled_trajectories, sampled_trajectories_data)
            )
            # sampled_trajectories_fine += sampled_trajectories_data.copy()

        # sampled_trajectories = np.vstack([sampled_trajectories, gt_trajectory[np.newaxis, :, :]])

        """
        # double check the sampled trajectories by plotting them
        trajectory_dir = f"./videos_new/once/{ref_scene_token}/logs_trajectory/"
        os.makedirs(trajectory_dir, exist_ok=True)
        trajectory_path = f"{trajectory_dir}/{ref_sd_token[1]}.jpg"
        # print(f"Doing {trajectory_path}:")
        for trajectory in sampled_trajectories:
            plt.plot(trajectory[:, 0], trajectory[:, 1])
        plt.grid(False)
        plt.axis("equal")
        plt.savefig(trajectory_path)
        plt.close()
        """

        #
        obj_boxes = self.load_object_boxes(ref_sd_token)
        obj_shadows = self.load_object_shadows(ref_sd_token)

        #
        fvf_maps = self.load_future_visible_freespace(ref_sd_token)

        drive_command = []

        #
        example = {
            "scene_token": ref_scene_token,
            "sample_data_token": ref_sd_token,
            "input_points": torch.from_numpy(np.concatenate(input_points_list)),
            "input_origin": torch.from_numpy(np.stack(input_origin_list)),
            "sampled_trajectories_fine": torch.from_numpy(sampled_trajectories_fine),
            "sampled_trajectories": torch.from_numpy(sampled_trajectories),
            "drive_command": torch.tensor(drive_command),
            "output_origin": torch.from_numpy(np.stack(output_origin_list)),
            "output_points": torch.from_numpy(np.concatenate(output_points_list)),
            "gt_trajectory": torch.from_numpy(gt_trajectory),
            "obj_boxes": torch.from_numpy(obj_boxes),
            "obj_shadows": torch.from_numpy(obj_shadows),
            "fvf_maps": torch.from_numpy(fvf_maps),
        }
        return example


class nuScenesDataset(Dataset):
    N_SWEEPS_PER_SAMPLE = 10
    SAMPLE_INTERVAL = 0.5  # second

    def __init__(self, nusc, nusc_split, kwargs, seed=0):
        super(nuScenesDataset, self).__init__()

        # set seed for split
        np.random.seed(seed)

        self.nusc = nusc
        self.nusc_root = self.nusc.dataroot
        self.nusc_can = NuScenesCanBus(dataroot=self.nusc_root)
        self.nusc_split = nusc_split

        # number of input samples
        self.n_input = kwargs["n_input"]

        # number of sampled trajectories
        self.n_samples = kwargs["n_samples"]

        # number of output samples
        self.n_output = kwargs["n_output"]
        assert self.n_output == 7

        #
        self.train_on_all_sweeps = kwargs["train_on_all_sweeps"]

        # scene-0419 does not have vehicle monitor data
        blacklist = [419] + self.nusc_can.can_blacklist

        # NOTE: use the official split (minus the ones in the blacklist)
        if "scene_token" in kwargs and kwargs["scene_token"] != "":
            scene = self.nusc.get("scene", kwargs["scene_token"])
            scenes = [scene]
        else:
            scene_splits = create_splits_scenes(verbose=False)
            scene_splits['train_2k'] = np.random.choice(scene_splits['train'], int(0.1 * len(scene_splits['train'])), replace=False)
            scene_splits['train_4k'] = np.random.choice(scene_splits['train'], int(0.2 * len(scene_splits['train'])), replace=False)
            scene_splits['train_8k'] = np.random.choice(scene_splits['train'], int(0.4 * len(scene_splits['train'])), replace=False)
            scene_splits['train_16k'] = np.random.choice(scene_splits['train'], int(0.8 * len(scene_splits['train'])), replace=False)
            with open('nusc_data_splits.pkl', 'wb') as f:
                pickle.dump(scene_splits, f)
            scene_names = scene_splits[self.nusc_split]
            scenes = []
            for scene in self.nusc.scene:
                scene_name = scene["name"]
                scene_no = int(scene_name[-4:])
                if (scene_name in scene_names) and (scene_no not in blacklist):
                    scenes.append(scene)

        # list all sample data
        self.valid_index = []
        self.flip_flags = []
        self.scene_tokens = []
        self.sample_data_tokens = []
        for scene in scenes:
            scene_token = scene["token"]
            # location
            log = self.nusc.get("log", scene["log_token"])
            # flip x axis if in left-hand traffic (singapore)
            flip_flag = True if log["location"].startswith("singapore") else False
            # record the token of every key frame
            start_index = len(self.sample_data_tokens)
            first_sample = self.nusc.get("sample", scene["first_sample_token"])
            sample_data_token = first_sample["data"]["LIDAR_TOP"]
            while sample_data_token != "":
                sample_data = self.nusc.get("sample_data", sample_data_token)
                if (self.nusc_split == "train" and self.train_on_all_sweeps) or (
                    sample_data["is_key_frame"]
                ):
                    self.flip_flags.append(flip_flag)
                    self.scene_tokens.append(scene_token)
                    self.sample_data_tokens.append(sample_data_token)
                sample_data_token = sample_data["next"]
            end_index = len(self.sample_data_tokens)
            # NOTE: make sure we have enough number of sweeps for input and output
            if self.nusc_split == "train" and self.train_on_all_sweeps:
                valid_start_index = start_index + self.n_input - 1
                valid_end_index = (
                    end_index - (self.n_output - 1) * self.N_SWEEPS_PER_SAMPLE
                )
            else:
                # NEW: acknowledge the fact and skip the first sample
                n_input_samples = self.n_input // self.N_SWEEPS_PER_SAMPLE
                valid_start_index = start_index + n_input_samples
                valid_end_index = end_index - self.n_output + 1
            self.valid_index += list(range(valid_start_index, valid_end_index))
        self._n_examples = len(self.valid_index)
        print(
            f"{self.nusc_split}: {self._n_examples} valid samples over {len(scenes)} scenes"
        )

    def __len__(self):
        return self._n_examples

    def get_global_pose(self, sd_token, inverse=False):
        sd = self.nusc.get("sample_data", sd_token)
        sd_ep = self.nusc.get("ego_pose", sd["ego_pose_token"])
        sd_cs = self.nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        if inverse is False:
            global_from_ego = transform_matrix(
                sd_ep["translation"], Quaternion(sd_ep["rotation"]), inverse=False
            )
            ego_from_sensor = transform_matrix(
                sd_cs["translation"], Quaternion(sd_cs["rotation"]), inverse=False
            )
            pose = global_from_ego.dot(ego_from_sensor)
        else:
            sensor_from_ego = transform_matrix(
                sd_cs["translation"], Quaternion(sd_cs["rotation"]), inverse=True
            )
            ego_from_global = transform_matrix(
                sd_ep["translation"], Quaternion(sd_ep["rotation"]), inverse=True
            )
            pose = sensor_from_ego.dot(ego_from_global)
        return pose

    def load_ground_segmentation(self, sample_data_token):
        path = f"{self.nusc.dataroot}/grndseg/{self.nusc.version}/{sample_data_token}_grndseg.bin"
        labels = np.fromfile(path, np.uint8)
        return labels

    def load_future_visible_freespace(self, sample_data_token):
        path = (
            f"{self.nusc.dataroot}/fvfmaps/{self.nusc.version}/{sample_data_token}.bin"
        )
        if os.path.exists(path):
            fvf_maps = np.fromfile(path, dtype=np.int8)
            fvf_maps = fvf_maps.reshape((7, 704, 400))
        else:
            fvf_maps = np.zeros((7, 704, 400), dtype=np.int8)
            # warnings.warn(f"Cannot find fvf_maps at {path}")
        return fvf_maps

    def load_object_boxes(self, sample_data_token):
        path = f"{self.nusc.dataroot}/obj_boxes/{self.nusc.version}/{sample_data_token}.bin"
        if os.path.exists(path):
            obj_boxes = np.fromfile(path, dtype=bool)
            obj_boxes = obj_boxes.reshape((7, 704, 400))
        else:
            obj_boxes = np.zeros((7, 704, 400))
        return obj_boxes

    def load_object_shadows(self, sample_data_token):
        path = f"{self.nusc.dataroot}/obj_shadows/{self.nusc.version}/{sample_data_token}.bin"
        if os.path.exists(path):
            obj_shadows = np.fromfile(path, dtype=bool)
            obj_shadows = obj_shadows.reshape((7, 704, 400))
        else:
            obj_shadows = np.zeros((7, 704, 400))
        return obj_shadows

    def __getitem__(self, idx):
        ref_index = self.valid_index[idx]

        ref_sd_token = self.sample_data_tokens[ref_index]
        ref_scene_token = self.scene_tokens[ref_index]
        flip_flag = self.flip_flags[ref_index]

        # reference coordinate frame
        ref_from_global = self.get_global_pose(ref_sd_token, inverse=True)

        # NOTE: input
        input_sds = []
        sd_token = ref_sd_token
        while sd_token != "" and len(input_sds) < self.n_input:
            curr_sd = self.nusc.get("sample_data", sd_token)
            input_sds.append(curr_sd)
            sd_token = curr_sd["prev"]

        # call out when we have less than the desired number of input sweeps
        # if len(input_sds) < self.n_input:
        #     warnings.warn(f"The number of input sweeps {len(input_sds)} is less than {self.n_input}.", RuntimeWarning)

        start = time.time()
        # get input sweep frames
        input_points_list = []
        input_origin_list = []
        for i, curr_sd in enumerate(input_sds):
            # load the current lidar sweep
            curr_lidar_pc = MyLidarPointCloud.from_file(
                f"{self.nusc_root}/{curr_sd['filename']}"
            )

            # remove ego returns
            curr_lidar_pc.remove_ego()

            # transform from the current lidar frame to global and then to the reference lidar frame
            global_from_curr = self.get_global_pose(curr_sd["token"], inverse=False)
            ref_from_curr = ref_from_global.dot(global_from_curr)
            curr_lidar_pc.transform(ref_from_curr)

            # NOTE: check if we are in Singapore (if so flip x)
            if flip_flag:
                curr_lidar_pc.points[0] *= -1
                ref_from_curr[0, 3] *= -1

            #
            origin = np.array(ref_from_curr[:3, 3])
            points = np.asarray(curr_lidar_pc.points[:3].T)
            tindex = np.full((len(points), 1), i)
            points = np.concatenate((points, tindex), axis=1)

            #
            input_points_list.append(points.astype(np.float32))
            input_origin_list.append(origin.astype(np.float32))


        # print("time to get input sweeps", time.time() - start)
        start = time.time()

        # NOTE: output
        # get output sample frames and ground truth trajectory
        output_origin_list = []
        output_points_list = []
        gt_trajectory = np.zeros((self.n_output, 3), np.float64)
        for i in range(self.n_output):
            if self.nusc_split == "train" and self.train_on_all_sweeps:
                index = ref_index + i * self.N_SWEEPS_PER_SAMPLE
            else:
                index = ref_index + i

            # if this exists a valid target
            if (
                index < len(self.scene_tokens)
                and self.scene_tokens[index] == ref_scene_token
            ):
                curr_sd_token = self.sample_data_tokens[index]
                curr_sd = self.nusc.get("sample_data", curr_sd_token)

                # load the current lidar sweep
                curr_lidar_pc = LidarPointCloud.from_file(
                    f"{self.nusc_root}/{curr_sd['filename']}"
                )

                # transform from the current lidar frame to global and then to the reference lidar frame
                global_from_curr = self.get_global_pose(curr_sd_token, inverse=False)
                ref_from_curr = ref_from_global.dot(global_from_curr)
                curr_lidar_pc.transform(ref_from_curr)

                #
                theta = quaternion_yaw(Quaternion(matrix=ref_from_curr))

                # NOTE: check if we are in Singapore (if so flip x)
                if flip_flag:
                    ref_from_curr[0, 3] *= -1
                    curr_lidar_pc.points[0] *= -1
                    theta *= -1

                origin = np.array(ref_from_curr[:3, 3])
                points = np.array(curr_lidar_pc.points[:3].T)
                gt_trajectory[i, :] = [origin[0], origin[1], theta]

                tindex = np.full(len(points), i)

                labels = self.load_ground_segmentation(curr_sd_token)
                assert len(labels) == len(points)
                mask = np.logical_and(labels >= 1, labels <= 30)

                points = np.concatenate(
                    (points, tindex[:, None], labels[:, None]), axis=1
                )
                points = points[mask, :]

            else:  # filler
                raise RuntimeError(f"The {i}-th output frame is not available")
                origin = np.array([0.0, 0.0, 0.0])
                points = np.full((0, 5), -1)

            # origin
            output_origin_list.append(origin.astype(np.float32))

            # points
            output_points_list.append(points.astype(np.float32))

        # print("time to get output sweeps", time.time() - start)
        start = time.time()

        # NOTE: trajectory sampling
        ref_scene = self.nusc.get("scene", ref_scene_token)

        # NOTE: rely on pose and steeranglefeedback data instead of vehicle_monitor
        vm_msgs = self.nusc_can.get_messages(ref_scene["name"], "vehicle_monitor")
        vm_uts = [msg["utime"] for msg in vm_msgs]
        pose_msgs = self.nusc_can.get_messages(ref_scene["name"], "pose")
        pose_uts = [msg["utime"] for msg in pose_msgs]
        steer_msgs = self.nusc_can.get_messages(ref_scene["name"], "steeranglefeedback")
        steer_uts = [msg["utime"] for msg in steer_msgs]

        # locate the closest message by universal timestamp
        ref_sd = self.nusc.get("sample_data", ref_sd_token)
        ref_utime = ref_sd["timestamp"]
        vm_index = locate_message(vm_uts, ref_utime)
        vm_data = vm_msgs[vm_index]
        pose_index = locate_message(pose_uts, ref_utime)
        pose_data = pose_msgs[pose_index]
        steer_index = locate_message(steer_uts, ref_utime)
        steer_data = steer_msgs[steer_index]

        # initial speed
        # v0 = vm_data["vehicle_speed"] / 3.6  # km/h to m/s
        v0 = pose_data["vel"][0]  # [0] means longitudinal velocity

        # curvature (positive: turn left)
        # steering = np.deg2rad(vm_data["steering"])
        steering = steer_data["value"]
        if flip_flag:
            steering *= -1

        ################################################################
        # override nuscenes values to test in the setting of once
        # steering = 0.0
        # v = (input_origin_list[0] - input_origin_list[1]) / 0.5
        # v0 = v[1] # [1] means longitudinal velocity (velocity along y)
        ################################################################

        # go ahead with gathering sampled trajectories
        Kappa = 2 * steering / 2.588

        #
        left_signal = vm_data["left_signal"]
        right_signal = vm_data["right_signal"]
        if flip_flag:
            left_signal, right_signal = right_signal, left_signal
        drive_command = [left_signal, right_signal]

        # initial state
        T0 = np.array([0.0, 1.0])  # define front
        N0 = (
            np.array([1.0, 0.0]) if Kappa <= 0 else np.array([-1.0, 0.0])
        )  # define side

        #
        # tt = np.arange(self.n_output) * self.SAMPLE_INTERVAL
        # tt = np.arange(0, self.n_output + self.SAMPLE_INTERVAL, self.SAMPLE_INTERVAL)
        t_start = 0  # second
        t_end = (self.n_output - 1) * self.SAMPLE_INTERVAL  # second
        t_interval = self.SAMPLE_INTERVAL / 10
        tt = np.arange(t_start, t_end + t_interval, t_interval)
        sampled_trajectories_fine = trajectory_sampler.sample(
            v0, Kappa, T0, N0, tt, self.n_samples
        )
        sampled_trajectories = sampled_trajectories_fine[:, ::10]

        # double check the sampled trajectories by plotting them
        # trajectory_dir = f"./videos_new/nusc/{ref_scene_token}/logs_trajectory/"
        # os.makedirs(trajectory_dir, exist_ok=True)
        # trajectory_path = f"{trajectory_dir}/{ref_sd_token}.jpg"
        # print(f"Doing {trajectory_path}:")
        # for trajectory in sampled_trajectories:
        #     plt.plot(trajectory[:, 0], trajectory[:, 1])
        # plt.grid(False)
        # plt.axis("equal")
        # plt.savefig(trajectory_path)
        # plt.close()

        # print("time to get samples trajectories", time.time() - start)
        start = time.time()

        #
        obj_boxes = self.load_object_boxes(ref_sd_token)
        obj_shadows = self.load_object_shadows(ref_sd_token)

        #
        fvf_maps = self.load_future_visible_freespace(ref_sd_token)

        # print("time to load everything else", time.time() - start)
        #
        example = {
            "scene_token": ref_scene_token,
            "sample_data_token": ref_sd_token,
            "input_points": torch.from_numpy(np.concatenate(input_points_list)),
            "sampled_trajectories_fine": torch.from_numpy(sampled_trajectories_fine),
            "sampled_trajectories": torch.from_numpy(sampled_trajectories),
            "drive_command": torch.tensor(drive_command),
            "output_origin": torch.from_numpy(np.stack(output_origin_list)),
            "output_points": torch.from_numpy(np.concatenate(output_points_list)),
            "gt_trajectory": torch.from_numpy(gt_trajectory),
            "obj_boxes": torch.from_numpy(obj_boxes),
            "obj_shadows": torch.from_numpy(obj_shadows),
            "fvf_maps": torch.from_numpy(fvf_maps),
        }
        return example


if __name__ == "__main__":
    pc_range = [-40.0, -70.4, -2.0, 40.0, 70.4, 3.4]
    voxel_size = 0.2
    n_input = 20
    n_samples = 1000
    n_output = 7
    dataset_kwargs = {
        "n_input": n_input,
        "n_samples": n_samples,
        "n_output": n_output,
        "train_on_all_sweeps": True,
    }
    torch.manual_seed(1)

    from nuscenes.nuscenes import NuScenes

    nusc = NuScenes("v1.0-mini", "/data3/tkhurana/datasets/nuScenes", verbose=True)
    # nusc = NuScenes("v1.0-trainval", "/data/nuscenes", verbose=True)

    dataset = nuScenesDataset(nusc, "train", dataset_kwargs)

    device = torch.device("cuda:0")

    from torch.utils.data import DataLoader

    data_loader_kwargs = {
        "pin_memory": False,
        "shuffle": True,
        "batch_size": 4,
        "num_workers": 0,
    }
    data_loader = DataLoader(dataset, collate_fn=CollateFn, **data_loader_kwargs)

    gt_trajs_dict = {0: [], 1: [], 2: [], 3: []}  # follow  # left  # right  # both
    for i, batch in enumerate(data_loader):
        import ipdb

        ipdb.set_trace()
        # _, in_pts, trajs, acts, out_orgs, out_pts, gt_trajs = batch
        # for j in range(len(acts)):
        #     a = (acts[j][0] + acts[j][1] * 2).item()
        #     gt_trajs_dict[a].append(gt_trajs[j].numpy())

    # for a in [0, 1, 2, 3]:
    #     if len(gt_trajs_dict[a]) > 0:
    #         gt_trajs_dict[a] = np.stack(gt_trajs_dict[a])

    # for a, name in zip([0, 1, 2, 3], ["follow", "left", "right", "both"]):
    #     gt_trajs = gt_trajs_dict[a]
    #     plt.clf()
    #     for i in range(len(gt_trajs)):
    #         plt.plot(gt_trajs[i, :, 0], gt_trajs[i, :, 1])
    #     plt.ylim([-70.4, 70.4])
    #     plt.xlim([-40., 40.])
    #     plt.savefig(f"gt_trajs_{name}.png")

    # from torch.utils.cpp_extension import load
    # raycaster = load("raycaster", sources=[
    #     "lib/raycaster.cpp", "lib/raycaster.cu"
    # ], verbose=True)

    # def transform(points, x_min=-40, y_min=-70.4, voxel_size=0.2):
    #     if points.ndim == 3:
    #         points[:, :, 0] = (points[:, :, 0] - x_min) / voxel_size
    #         points[:, :, 1] = (points[:, :, 1] - y_min) / voxel_size
    #     elif points.ndim == 4:  # trajectories
    #         points[:, :, :, 0] = (points[:, :, :, 0] - x_min) / voxel_size
    #         points[:, :, :, 1] = (points[:, :, :, 1] - y_min) / voxel_size
    #     return points

    # for i, batch in enumerate(data_loader):
    #     filenames, input_points, sampled_trajectories, output_origins, output_points, gt_trajectories = batch

    #     input_points = transform(input_points)
    #     sampled_trajectories = transform(sampled_trajectories)
    #     output_origins = transform(output_origins)
    #     output_points = transform(output_points)
    #     gt_trajectories = transform(gt_trajectories)

    #     output_origins = output_origins.to(device)
    #     output_points = output_points.to(device)

    #     # t x y
    #     # stopper, occupancy = raycaster.raycast(output_origins, output_points, [n_output+1, 704, 400])
    #     occupancy = raycaster.raycast(output_origins, output_points, [n_output, 704, 400])
    #     # stopper = stopper.detach().cpu().numpy()
    #     occupancy = occupancy.detach().cpu().numpy()

    #     sampled_trajectories = sampled_trajectories.detach().cpu().numpy()
    #     gt_trajectories = gt_trajectories.detach().cpu().numpy()

    #     for j in range(len(occupancy)):
    #         print(i, j, output_points[j].shape)
    #         for k in range(len(sampled_trajectories[j])):
    #             xs = sampled_trajectories[j, k, :, 0].astype(int)
    #             ys = sampled_trajectories[j, k, :, 1].astype(int)
    #             thetas = sampled_trajectories[j, k, :, 2]
    #             valid = np.logical_and(
    #                 np.logical_and(xs >= 0, xs < 400),
    #                 np.logical_and(ys >= 0, ys < 704),
    #             )
    #             xs = xs[valid]
    #             ys = ys[valid]
    #             thetas = thetas[valid]
    #             occupancy[j, :, ys, xs] = thetas[:, None]

    #         for t in range(n_output):
    #             image_path = f"sanity_checks/{i}_{j}_{t}.png"
    #             x = int(gt_trajectories[j][t][0])
    #             y = int(gt_trajectories[j][t][1])
    #             theta = gt_trajectories[j][t][2]
    #             W = 2
    #             if y >= 0+W and y < 704-W and x >= 0+W and x < 400-W:
    #                 occupancy[j, t, y-W:y+W+1, x-W:x+W+1] = theta

    #             # plt.imsave(image_path, np.concatenate((stopper[j][t], occupancy[j][t]), axis=1))
    #             plt.imsave(image_path, occupancy[j][t][::-1])

    #     if i >= 10:
    #         break
