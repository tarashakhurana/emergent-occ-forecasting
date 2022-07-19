import abc
import numpy as np

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

# from skimage.draw import polygon
from scipy import ndimage

# JIT
from torch.utils.cpp_extension import load

raycaster = load(
    "raycaster",
    sources=["lib/raycast/raycaster.cpp", "lib/raycast/raycaster.cu"],
    verbose=True,
)
voxelizer = load(
    "voxelizer",
    sources=["lib/voxelize/voxelizer.cpp", "lib/voxelize/voxelizer.cu"],
    verbose=True,
)
renderer = load(
    "renderer",
    sources=["lib/render/renderer.cpp", "lib/render/renderer.cu"],
    verbose=True,
)
raymaxer = load(
    "raymaxer",
    sources=["lib/raymax/raymaxer.cpp", "lib/raymax/raymaxer.cu"],
    verbose=True,
)


def conv3x3(in_channels, out_channels, bias=False):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias
    )


def deconv3x3(in_channels, out_channels, stride):
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        output_padding=1,
        bias=False,
    )


def maxpool2x2(stride):
    return nn.MaxPool2d(kernel_size=2, stride=stride, padding=0)


def relu(inplace=True):
    return nn.ReLU(inplace=inplace)


def bn(num_features):
    return nn.BatchNorm2d(num_features=num_features)


class ConvBlock(nn.Module):
    def __init__(self, num_layer, in_channels, out_channels, max_pool=False):
        super(ConvBlock, self).__init__()

        layers = []
        for i in range(num_layer):
            _in_channels = in_channels if i == 0 else out_channels
            layers.append(conv3x3(_in_channels, out_channels))
            layers.append(bn(out_channels))
            layers.append(relu())

        if max_pool:
            layers.append(maxpool2x2(stride=2))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, num_layers, num_filters):
        super(Encoder, self).__init__()

        self.in_channels = in_channels
        self.out_channels = num_filters[4]

        # Block 1-4
        _in_channels = self.in_channels
        self.block1 = ConvBlock(
            num_layers[0], _in_channels, num_filters[0], max_pool=True
        )
        self.block2 = ConvBlock(
            num_layers[1], num_filters[0], num_filters[1], max_pool=True
        )
        self.block3 = ConvBlock(
            num_layers[2], num_filters[1], num_filters[2], max_pool=True
        )
        self.block4 = ConvBlock(num_layers[3], num_filters[2], num_filters[3])

        # Block 5
        _in_channels = sum(num_filters[0:4])
        self.block5 = ConvBlock(num_layers[4], _in_channels, num_filters[4])

    def forward(self, x):
        N, C, H, W = x.shape

        # the first 4 blocks
        c1 = self.block1(x)
        c2 = self.block2(c1)
        c3 = self.block3(c2)
        c4 = self.block4(c3)

        # upsample and concat
        _H, _W = H // 4, W // 4
        c1_interp = F.interpolate(
            c1, size=(_H, _W), mode="bilinear", align_corners=True
        )
        c2_interp = F.interpolate(
            c2, size=(_H, _W), mode="bilinear", align_corners=True
        )
        c3_interp = F.interpolate(
            c3, size=(_H, _W), mode="bilinear", align_corners=True
        )
        c4_interp = F.interpolate(
            c4, size=(_H, _W), mode="bilinear", align_corners=True
        )

        #
        c4_aggr = torch.cat((c1_interp, c2_interp, c3_interp, c4_interp), dim=1)
        c5 = self.block5(c4_aggr)

        return c5


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block = nn.Sequential(
            deconv3x3(in_channels, 128, stride=2),
            bn(128),
            relu(),
            conv3x3(128, 128),
            bn(128),
            relu(),
            deconv3x3(128, 64, stride=2),
            bn(64),
            relu(),
            conv3x3(64, 64),
            bn(64),
            relu(),
            conv3x3(64, out_channels, bias=True),
        )

    def forward(self, x):
        return self.block(x)


class BaseNeuralMotionPlanner(nn.Module):
    MAX_COST = 1000.0

    def __init__(self, n_input, n_output, pc_range, voxel_size):

        super(BaseNeuralMotionPlanner, self).__init__()

        self.n_input = n_input
        self.n_output = n_output

        self.n_height = int((pc_range[5] - pc_range[2]) / voxel_size)
        self.n_length = int((pc_range[4] - pc_range[1]) / voxel_size)
        self.n_width = int((pc_range[3] - pc_range[0]) / voxel_size)

        self.input_grid = [self.n_input, self.n_height, self.n_length, self.n_width]
        self.output_grid = [self.n_output, self.n_length, self.n_width]

        self.pc_range = pc_range
        self.voxel_size = voxel_size

        self.offset = torch.nn.parameter.Parameter(
            torch.Tensor(self.pc_range[:3])[None, None, :], requires_grad=False
        )
        self.scaler = torch.nn.parameter.Parameter(
            torch.Tensor([self.voxel_size] * 3)[None, None, :], requires_grad=False
        )

        self._in_channels = self.n_input * self.n_height
        self.encoder = Encoder(
            self._in_channels, [2, 2, 3, 6, 5], [32, 64, 128, 256, 256]
        )

        # NOTE: initialize the linear predictor (no bias) over history
        self._out_channels = self.n_output
        self.imitation_decoder = Decoder(self.encoder.out_channels, self._out_channels)

    def _compute_L2(self, batch):
        # 1000 sampled trajectories vs 1 gt trajectory
        st, gt = batch["sampled_trajectories"], batch["gt_trajectories"]

        # L2 distance
        return torch.sqrt(((st[:, :, :, :2] - gt[:, None, :, :2]) ** 2).sum(dim=-1))

    def _normalize(self, points):
        points[:, :, :3] = (points[:, :, :3] - self.offset) / self.scaler

    def _discretize(self, trajectories):
        # input: N x n_samples x n_output x 3 (x, y, theta)
        # output: N x n_samples x n_output (yi, xi)
        if trajectories.ndim == 3:  # ground truth trajectories
            trajectories = trajectories[:, None, :, :]

        #
        N, M, T, _ = trajectories.shape

        #
        xx, yy = trajectories[:, :, :, 0], trajectories[:, :, :, 1]

        # discretize
        yi = ((yy - self.pc_range[1]) / self.voxel_size).long()
        yi = torch.clamp(yi, min=0, max=self.n_length - 1)

        xi = ((xx - self.pc_range[0]) / self.voxel_size).long()
        xi = torch.clamp(xi, min=0, max=self.n_width - 1)

        #
        return yi, xi

    def prepare_input(self, batch):
        # extract data
        input_points = batch["input_points"]

        # convert metric coordinates to grid coordinates
        self._normalize(input_points)

        # voxelize input LiDAR sweeps
        input_tensor = voxelizer.voxelize(input_points, self.input_grid)
        input_tensor = input_tensor.reshape(
            (-1, self._in_channels, self.n_length, self.n_width)
        )

        return input_tensor

    def compute_cost_maps(self, feat):
        return

    def clamp_cost_maps(self, C):
        return torch.clamp(C, min=-self.MAX_COST, max=self.MAX_COST)

    def evaluate_samples(self, batch, C):
        # parse input
        sampled_trajectories = batch["sampled_trajectories"]

        # batch size
        N = len(sampled_trajectories)
        ii = torch.arange(N)
        ti = torch.arange(self.n_output)

        # discretize trajectories
        Syi, Sxi = self._discretize(sampled_trajectories)

        # indexing
        CS = C[ii[:, None, None], ti[None, None, :], Syi, Sxi]

        #
        return CS

    def evaluate_expert(self, batch, C):
        # parse input
        gt_trajectories = batch["gt_trajectories"]

        # batch size
        N = len(gt_trajectories)
        ii = torch.arange(N)
        ti = torch.arange(self.n_output)

        # discretize trajectories
        Gyi, Gxi = self._discretize(gt_trajectories)

        # indexing
        CG = C[ii[:, None, None], ti[None, None, :], Gyi, Gxi]

        #
        return CG

    @abc.abstractmethod
    def compute_cost_margins(self, batch):
        return

    def select_best_plans(self, batch, CS, k=1):
        sampled_trajectories = batch["sampled_trajectories"]

        #
        N = len(sampled_trajectories)
        ii = torch.arange(N)

        # select top 5 (lowest) cost trajectories
        CC, KK = torch.topk(CS.sum(-1), k, dim=-1, largest=False)

        #
        return KK

    def select_best_plans_given_gt(self, batch, k=1):
        # select top 5 (lowest) cost trajectories
        l2 = self._compute_L2(batch).sum(-1)
        CC, KK = torch.topk(l2, k, dim=-1, largest=False)

        #
        return KK

    def forward(self, batch, mode="train"):
        results = {}

        # voxelize input lidar sweeps
        I = self.prepare_input(batch)

        # extract backbone feature maps
        feat = self.encoder(I)

        # compute cost maps (model-specific)
        C = self.compute_cost_maps(feat)

        # clamp cost
        C = self.clamp_cost_maps(C)

        # evaluate the cost of every sampled trajectory
        CS = self.evaluate_samples(batch, C)

        if mode == "train":
            # evaluate the cost of the expert trajectory
            CG = self.evaluate_expert(batch, C)

            # compute cost margins (model-specific)
            CM = self.compute_cost_margins(batch)

            # construct the max-margin loss
            L, _ = ((F.relu(CG - CS + CM)).sum(dim=-1)).max(dim=-1)

            # return the margin loss
            results["margin_loss"] = L

        else:
            results["il_cost"] = C
            results["cost"] = C
            results["best_plans"] = self.select_best_plans(batch, CS, 5)

        return results


class VanillaNeuralMotionPlanner(BaseNeuralMotionPlanner):
    def __init__(self, n_input, n_output, pc_range, voxel_size):
        super(VanillaNeuralMotionPlanner, self).__init__(
            n_input, n_output, pc_range, voxel_size
        )

    def compute_cost_margins(self, batch):
        return self._compute_L2(batch)

    def compute_cost_maps(self, feat):
        return self.imitation_decoder(feat)

    def forward(self, batch, mode):
        results = super(VanillaNeuralMotionPlanner, self).forward(batch, mode)
        if mode == "train":
            results["loss"] = results["margin_loss"]
        return results


class VFGuidedNeuralMotionPlanner(BaseNeuralMotionPlanner):
    NVF_COST_FACTOR = 200.0

    def __init__(self, n_input, n_output, pc_range, voxel_size):
        super(VFGuidedNeuralMotionPlanner, self).__init__(
            n_input, n_output, pc_range, voxel_size
        )

    def compute_cost_margins(self, batch):
        # incorporate visible freespace as part of the cost margin
        # output_origins = batch["output_origins"]
        # output_points = batch["output_points"]
        # self._normalize(output_origins)
        # self._normalize(output_points)
        # freespace = raycaster.raycast(output_origins, output_points, self.output_grid)
        freespace = batch["fvf_maps"]

        # discretize sampled trajectories
        sampled_trajectories = batch["sampled_trajectories"]
        N = len(sampled_trajectories)
        ii = torch.arange(N)
        ti = torch.arange(self.n_output)
        Syi, Sxi = self._discretize(sampled_trajectories)

        # index observed future visible freespace with sampled trajectories
        # observed freespace is marked as -1
        label = freespace[ii[:, None, None], ti[None, None, :], Syi, Sxi]

        #
        nvf_cost = self.NVF_COST_FACTOR * (label != -1).float()

        #
        l2_cost = self._compute_L2(batch)

        return nvf_cost + l2_cost

    def compute_cost_maps(self, feat):
        return self.imitation_decoder(feat)

    def forward(self, batch, mode):
        results = super(VFGuidedNeuralMotionPlanner, self).forward(batch, mode)
        if mode == "train":
            results["loss"] = results["margin_loss"]
        return results


class ObjGuidedNeuralMotionPlanner(BaseNeuralMotionPlanner):
    OBJ_COST_FACTOR = 200.0

    def __init__(self, n_input, n_output, pc_range, voxel_size):
        super(ObjGuidedNeuralMotionPlanner, self).__init__(
            n_input, n_output, pc_range, voxel_size
        )

    def compute_cost_margins(self, batch):
        # incorporate visible freespace as part of the cost margin
        obj_maps = batch["obj_boxes"]

        # discretize sampled trajectories
        sampled_trajectories = batch["sampled_trajectories"]
        N = len(sampled_trajectories)
        ii = torch.arange(N)
        ti = torch.arange(self.n_output)
        Syi, Sxi = self._discretize(sampled_trajectories)

        # index observed future visible freespace with sampled trajectories
        # observed freespace is marked as -1
        label = obj_maps[ii[:, None, None], ti[None, None, :], Syi, Sxi]

        #
        obj_cost = self.OBJ_COST_FACTOR * (label == 1).float()

        #
        l2_cost = self._compute_L2(batch)

        return obj_cost + l2_cost

    def compute_cost_maps(self, feat):
        return self.imitation_decoder(feat)

    def forward(self, batch, mode):
        results = super(ObjGuidedNeuralMotionPlanner, self).forward(batch, mode)
        if mode == "train":
            results["loss"] = results["margin_loss"]
        return results


class ObjShadowGuidedNeuralMotionPlanner(BaseNeuralMotionPlanner):
    OBJ_COST_FACTOR = 200.0

    def __init__(self, n_input, n_output, pc_range, voxel_size):
        super(ObjShadowGuidedNeuralMotionPlanner, self).__init__(
            n_input, n_output, pc_range, voxel_size
        )

    def compute_cost_margins(self, batch):
        # incorporate visible freespace as part of the cost margin
        # obj_maps = batch["obj_maps"]
        obj_maps = batch["obj_shadows"]

        # discretize sampled trajectories
        sampled_trajectories = batch["sampled_trajectories"]
        N = len(sampled_trajectories)
        ii = torch.arange(N)
        ti = torch.arange(self.n_output)
        Syi, Sxi = self._discretize(sampled_trajectories)

        # index observed future visible freespace with sampled trajectories
        # observed freespace is marked as -1
        label = obj_maps[ii[:, None, None], ti[None, None, :], Syi, Sxi]

        #
        obj_cost = self.OBJ_COST_FACTOR * (label == 1).float()

        #
        l2_cost = self._compute_L2(batch)

        return obj_cost + l2_cost

    def compute_cost_maps(self, feat):
        return self.imitation_decoder(feat)

    def forward(self, batch, mode):
        results = super(ObjShadowGuidedNeuralMotionPlanner, self).forward(batch, mode)
        if mode == "train":
            results["loss"] = results["margin_loss"]
        return results


class VFExplicitNeuralMotionPlanner(BaseNeuralMotionPlanner):
    NVF_COST_FACTOR = 200.0

    def __init__(self, n_input, n_output, pc_range, voxel_size, nvf_loss_factor=1.0):
        super(VFExplicitNeuralMotionPlanner, self).__init__(
            n_input, n_output, pc_range, voxel_size
        )
        # an additional decoder for predicting future visible freespace
        self.nvf_decoder = Decoder(self.encoder.out_channels, self._out_channels)
        self.nvf_loss_factor = nvf_loss_factor

    def compute_cost_margins(self, batch):
        return self._compute_L2(batch)

    def compute_nvf_target(self, batch):
        # incorporate visible freespace as part of the cost margin
        output_origins = batch["output_origins"]
        output_points = batch["output_points"]
        self._normalize(output_origins)
        self._normalize(output_points)
        freespace = raycaster.raycast(output_origins, output_points, self.output_grid)
        # positive: non visible freespace
        target = freespace != -1
        return target

    def forward(self, batch, mode):
        results = {}

        # voxelize input lidar sweeps
        I = self.prepare_input(batch)

        # extract backbone feature maps
        feat = self.encoder(I)

        # compute cost maps
        # imitation learning component
        il_cost = self.imitation_decoder(feat)

        # non-freespace component
        nvf_logits = self.nvf_decoder(feat)
        nvf_cost = torch.sigmoid(nvf_logits) * self.NVF_COST_FACTOR

        # final cost map is a combination of two
        C = il_cost + nvf_cost

        # clamp cost
        C = self.clamp_cost_maps(C)

        # evaluate the cost of every sampled trajectory
        CS = self.evaluate_samples(batch, C)

        if mode == "train":
            # evaluate the cost of the expert trajectory
            CG = self.evaluate_expert(batch, C)

            # compute cost margins (model-specific)
            CM = self.compute_cost_margins(batch)

            # construct the max-margin loss
            Lm, _ = ((F.relu(CG - CS + CM)).sum(dim=-1)).max(dim=-1)

            # return the margin loss
            results["margin_loss"] = Lm

            # second part of the loss: visible freespace classification
            nvf_target = self.compute_nvf_target(batch).float()

            # binary cross entropy loss
            Lf = F.binary_cross_entropy_with_logits(nvf_logits, nvf_target)

            #
            results["nvf_loss"] = Lf

            #
            results["loss"] = Lm + Lf * self.nvf_loss_factor

        else:
            results["il_cost"] = il_cost
            results["nvf_cost"] = nvf_cost
            results["nvf_prob"] = torch.sigmoid(nvf_logits)
            results["cost"] = C
            results["best_plans"] = self.select_best_plans(batch, CS, 5)

        return results


class VFSupervisedNeuralMotionPlanner(BaseNeuralMotionPlanner):
    NVF_COST_FACTOR = 200.0

    def __init__(self, n_input, n_output, pc_range, voxel_size, nvf_loss_factor=1.0):
        super(VFSupervisedNeuralMotionPlanner, self).__init__(
            n_input, n_output, pc_range, voxel_size
        )
        # an additional decoder for predicting future visible freespace
        self.nvf_decoder = Decoder(self.encoder.out_channels, self._out_channels)
        self.nvf_loss_factor = nvf_loss_factor
        print(f"VFSupervisedNeuralMotionPlanner: nvf loss factor is {nvf_loss_factor}")

    def compute_cost_margins(self, batch):
        # incorporate visible freespace as part of the cost margin
        # output_origins = batch["output_origins"]
        # output_points = batch["output_points"]
        # self._normalize(output_origins)
        # self._normalize(output_points)
        # freespace = raycaster.raycast(output_origins, output_points, self.output_grid)
        # freespace2 = batch["fvf_maps"]
        # freespace_np = freespace.detach().cpu().numpy()
        # freespace2_np = freespace2.detach().cpu().numpy()
        # import matplotlib.pyplot as plt
        # plt.imshow(np.concatenate((
        #     np.concatenate(freespace_np[0], axis=-1)[::-1],
        #     np.concatenate(freespace2_np[0], axis=-1)[::-1],
        # ), axis=0))
        # import ipdb
        # ipdb.set_trace()
        freespace = batch["fvf_maps"]

        # discretize sampled trajectories
        sampled_trajectories = batch["sampled_trajectories"]
        N = len(sampled_trajectories)
        ii = torch.arange(N)
        ti = torch.arange(self.n_output)
        Syi, Sxi = self._discretize(sampled_trajectories)

        # index observed future visible freespace with sampled trajectories
        # observed freespace is marked as -1
        label = freespace[ii[:, None, None], ti[None, None, :], Syi, Sxi]

        #
        nvf_cost = self.NVF_COST_FACTOR * (label != -1).float()

        #
        l2_cost = self._compute_L2(batch)

        return nvf_cost + l2_cost

    def compute_nvf_target(self, batch):
        # incorporate visible freespace as part of the cost margin
        # output_origins = batch["output_origins"]
        # output_points = batch["output_points"]
        # self._normalize(output_origins)
        # self._normalize(output_points)
        # freespace = raycaster.raycast(output_origins, output_points, self.output_grid)
        freespace = batch["fvf_maps"]

        # positive: non visible freespace
        target = freespace != -1
        return target

    def forward(self, batch, mode):
        results = {}

        # voxelize input lidar sweeps
        I = self.prepare_input(batch)

        # extract backbone feature maps
        feat = self.encoder(I)

        # compute cost maps
        # imitation learning component
        il_cost = self.imitation_decoder(feat)

        # non-freespace component
        nvf_logits = self.nvf_decoder(feat)
        nvf_cost = torch.sigmoid(nvf_logits) * self.NVF_COST_FACTOR

        # final cost map is a combination of two
        C = il_cost + nvf_cost

        # clamp cost
        C = self.clamp_cost_maps(C)

        # evaluate the cost of every sampled trajectory
        CS = self.evaluate_samples(batch, C)

        if mode == "train":
            # evaluate the cost of the expert trajectory
            CG = self.evaluate_expert(batch, C)

            # compute cost margins (model-specific)
            CM = self.compute_cost_margins(batch)

            # construct the max-margin loss
            Lm, _ = ((F.relu(CG - CS + CM)).sum(dim=-1)).max(dim=-1)

            # return the margin loss
            results["margin_loss"] = Lm

            # second part of the loss: visible freespace classification
            nvf_target = self.compute_nvf_target(batch).float()

            # binary cross entropy loss
            Lf = F.binary_cross_entropy_with_logits(nvf_logits, nvf_target)

            #
            results["nvf_loss"] = Lf

            #
            results["loss"] = Lm + Lf * self.nvf_loss_factor

        else:
            results["il_cost"] = il_cost
            results["nvf_prob"] = torch.sigmoid(nvf_logits)
            results["nvf_cost"] = nvf_cost
            results["cost"] = C
            results["best_plans"] = self.select_best_plans(batch, CS, 5)

        return results


class ObjExplicitNeuralMotionPlanner(BaseNeuralMotionPlanner):
    OBJ_COST_FACTOR = 200.0

    def __init__(self, n_input, n_output, pc_range, voxel_size, obj_loss_factor=1.0):
        super(ObjExplicitNeuralMotionPlanner, self).__init__(
            n_input, n_output, pc_range, voxel_size
        )
        # an additional decoder for predicting future visible freespace
        self.obj_decoder = Decoder(self.encoder.out_channels, self._out_channels)
        self.obj_loss_factor = obj_loss_factor

    def compute_cost_margins(self, batch):
        return self._compute_L2(batch)

    def compute_obj_target(self, batch):
        # incorporate visible freespace as part of the cost margin
        # obj_maps = batch["obj_maps"]
        obj_maps = batch["obj_boxes"]
        target = (obj_maps == 1).float()
        return target

    def forward(self, batch, mode):
        results = {}

        # voxelize input lidar sweeps
        I = self.prepare_input(batch)

        # extract backbone feature maps
        feat = self.encoder(I)

        # compute cost maps
        # imitation learning component
        il_cost = self.imitation_decoder(feat)

        # non-freespace component
        obj_logits = self.obj_decoder(feat)
        obj_cost = torch.sigmoid(obj_logits) * self.OBJ_COST_FACTOR

        # final cost map is a combination of two
        C = il_cost + obj_cost

        # clamp cost
        C = self.clamp_cost_maps(C)

        # evaluate the cost of every sampled trajectory
        CS = self.evaluate_samples(batch, C)

        if mode == "train":
            # evaluate the cost of the expert trajectory
            CG = self.evaluate_expert(batch, C)

            # compute cost margins (model-specific)
            CM = self.compute_cost_margins(batch)

            # construct the max-margin loss
            Lm, _ = ((F.relu(CG - CS + CM)).sum(dim=-1)).max(dim=-1)

            # return the margin loss
            results["margin_loss"] = Lm

            # second part of the loss: visible freespace classification
            obj_target = self.compute_obj_target(batch).float()

            # binary cross entropy loss
            Lf = F.binary_cross_entropy_with_logits(obj_logits, obj_target)

            #
            results["obj_loss"] = Lf

            #
            results["loss"] = Lm + Lf * self.obj_loss_factor

        else:
            results["il_cost"] = il_cost
            results["obj_cost"] = obj_cost
            results["cost"] = C
            results["best_plans"] = self.select_best_plans(batch, CS, 5)

        return results


class ObjSupervisedNeuralMotionPlanner(BaseNeuralMotionPlanner):
    OBJ_COST_FACTOR = 200.0

    def __init__(self, n_input, n_output, pc_range, voxel_size, obj_loss_factor=1.0):
        super(ObjSupervisedNeuralMotionPlanner, self).__init__(
            n_input, n_output, pc_range, voxel_size
        )
        # an additional decoder for predicting future visible freespace
        self.obj_decoder = Decoder(self.encoder.out_channels, self._out_channels)
        self.obj_loss_factor = obj_loss_factor
        print("ObjSupervisedNeuralMotionPlanner: obj loss factor is", obj_loss_factor)

    def compute_cost_margins(self, batch):
        # incorporate visible freespace as part of the cost margin
        # obj_maps = batch["obj_maps"]
        obj_maps = batch["obj_boxes"]

        # discretize sampled trajectories
        sampled_trajectories = batch["sampled_trajectories"]
        N = len(sampled_trajectories)
        ii = torch.arange(N)
        ti = torch.arange(self.n_output)
        Syi, Sxi = self._discretize(sampled_trajectories)

        # index observed future visible freespace with sampled trajectories
        # observed freespace is marked as -1
        label = obj_maps[ii[:, None, None], ti[None, None, :], Syi, Sxi]

        #
        obj_cost = self.OBJ_COST_FACTOR * (label == 1).float()

        #
        l2_cost = self._compute_L2(batch)

        return obj_cost + l2_cost

    def compute_obj_target(self, batch):
        # incorporate visible freespace as part of the cost margin
        # obj_maps = batch["obj_maps"]
        obj_maps = batch["obj_boxes"]
        target = (obj_maps == 1).float()
        return target

    def forward(self, batch, mode):
        results = {}

        # voxelize input lidar sweeps
        I = self.prepare_input(batch)

        # extract backbone feature maps
        feat = self.encoder(I)

        # compute cost maps
        # imitation learning component
        il_cost = self.imitation_decoder(feat)

        # non-freespace component
        obj_logits = self.obj_decoder(feat)
        obj_cost = torch.sigmoid(obj_logits) * self.OBJ_COST_FACTOR

        # final cost map is a combination of two
        C = il_cost + obj_cost

        # clamp cost
        C = self.clamp_cost_maps(C)

        # evaluate the cost of every sampled trajectory
        CS = self.evaluate_samples(batch, C)

        if mode == "train":
            # evaluate the cost of the expert trajectory
            CG = self.evaluate_expert(batch, C)

            # compute cost margins (model-specific)
            CM = self.compute_cost_margins(batch)

            # construct the max-margin loss
            Lm, _ = ((F.relu(CG - CS + CM)).sum(dim=-1)).max(dim=-1)

            # return the margin loss
            results["margin_loss"] = Lm

            # second part of the loss: visible freespace classification
            obj_target = self.compute_obj_target(batch).float()

            # binary cross entropy loss
            Lf = F.binary_cross_entropy_with_logits(obj_logits, obj_target)

            #
            results["obj_loss"] = Lf

            #
            results["loss"] = Lm + Lf * self.obj_loss_factor

        else:
            results["il_cost"] = il_cost
            results["obj_prob"] = torch.sigmoid(obj_logits)
            results["obj_cost"] = obj_cost
            results["cost"] = C
            results["best_plans"] = self.select_best_plans(batch, CS, 5)

        return results


class ObjShadowSupervisedNeuralMotionPlanner(ObjSupervisedNeuralMotionPlanner):
    OBJ_COST_FACTOR = 200.0

    def __init__(self, n_input, n_output, pc_range, voxel_size, obj_loss_factor=1.0):
        super(ObjShadowSupervisedNeuralMotionPlanner, self).__init__(
            n_input, n_output, pc_range, voxel_size, obj_loss_factor=obj_loss_factor
        )
        print(
            "ObjShadowSupervisedNeuralMotionPlanner: obj loss factor is",
            obj_loss_factor,
        )

    def compute_cost_margins(self, batch):
        # incorporate visible freespace as part of the cost margin
        obj_maps = batch["obj_shadows"]

        # discretize sampled trajectories
        sampled_trajectories = batch["sampled_trajectories"]
        N = len(sampled_trajectories)
        ii = torch.arange(N)
        ti = torch.arange(self.n_output)
        Syi, Sxi = self._discretize(sampled_trajectories)

        # index observed future visible freespace with sampled trajectories
        # observed freespace is marked as -1
        label = obj_maps[ii[:, None, None], ti[None, None, :], Syi, Sxi]

        #
        obj_cost = self.OBJ_COST_FACTOR * (label == 1).float()

        #
        l2_cost = self._compute_L2(batch)

        return obj_cost + l2_cost

    def compute_obj_target(self, batch):
        # incorporate visible freespace as part of the cost margin
        obj_maps = batch["obj_shadows"]
        target = (obj_maps == 1).float()
        return target

    def forward(self, batch, mode):
        results = super(ObjShadowSupervisedNeuralMotionPlanner, self).forward(
            batch, mode
        )
        return results


class OccExplicitNeuralMotionPlanner(BaseNeuralMotionPlanner):
    OCC_COST_FACTOR = 200.0

    def __init__(self, n_input, n_output, pc_range, voxel_size, occ_loss_factor=1.0):
        super(OccExplicitNeuralMotionPlanner, self).__init__(
            n_input, n_output, pc_range, voxel_size
        )
        # an additional decoder for predicting future visible freespace
        self.occ_decoder = Decoder(self.encoder.out_channels, self._out_channels)
        self.occ_loss_factor = occ_loss_factor
        print("occ loss factor is", self.occ_loss_factor)

    def compute_cost_margins(self, batch):
        return self._compute_L2(batch)

    def compute_occ_target(self, batch):
        # incorporate visible freespace as part of the cost margin
        output_origins = batch["output_origins"]
        output_points = batch["output_points"]
        self._normalize(output_origins)
        self._normalize(output_points)
        freespace = raycaster.raycast(output_origins, output_points, self.output_grid)
        # positive: non visible freespace
        target = (freespace == 1).float()
        valid = freespace != 0
        return target, valid

    def forward(self, batch, mode):
        results = {}

        # voxelize input lidar sweeps
        I = self.prepare_input(batch)

        # extract backbone feature maps
        feat = self.encoder(I)

        # compute cost maps
        # imitation learning component
        il_cost = self.imitation_decoder(feat)

        # non-freespace component
        occ_logits = self.occ_decoder(feat)
        occ_cost = torch.sigmoid(occ_logits) * self.OCC_COST_FACTOR

        # final cost map is a combination of two
        C = il_cost + occ_cost

        # clamp cost
        C = self.clamp_cost_maps(C)

        # evaluate the cost of every sampled trajectory
        CS = self.evaluate_samples(batch, C)

        if mode == "train":
            # evaluate the cost of the expert trajectory
            CG = self.evaluate_expert(batch, C)

            # compute cost margins (model-specific)
            CM = self.compute_cost_margins(batch)

            # construct the max-margin loss
            Lm, _ = ((F.relu(CG - CS + CM)).sum(dim=-1)).max(dim=-1)

            # return the margin loss
            results["margin_loss"] = Lm

            # second part of the loss: visible freespace classification
            occ_target, occ_valid = self.compute_occ_target(batch)

            # binary cross entropy loss
            Lfs = F.binary_cross_entropy_with_logits(
                occ_logits, occ_target, reduction="none"
            )
            Lf = Lfs[occ_valid].sum() / occ_valid.sum()

            #
            results["occ_loss"] = Lf

            #
            results["loss"] = Lm + Lf * self.occ_loss_factor

        else:
            results["il_cost"] = il_cost
            results["occ_cost"] = occ_cost
            results["cost"] = C
            results["best_plans"] = self.select_best_plans(batch, CS, 5)

        return results


class DifferentiableRenderingLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sigma, origins, points):
        # differentiable rendering
        # pred_dist, gt_dist, grad_sigma = renderer.render(sigma, origins, points)
        losses, grad_sigma = renderer.render(sigma, origins, points)

        # compute the rendering loss: L1 distance
        # losses = torch.abs(pred_dist - gt_dist)

        # valid examples have non-negative distance
        # compute the average loss per ray
        mask = losses >= 0
        count = mask.sum()
        loss = losses[mask].sum() / count

        # cache the gradients we computed for the backward pass
        # scale gradient in the same way we average the loss across all rays
        grad_sigma /= count
        ctx.save_for_backward(grad_sigma)

        return loss

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output is occ_loss_factor
        # extract the cached gradient tensor
        (grad_sigma,) = ctx.saved_tensors
        # pass the gradients on
        return grad_sigma * grad_output, None, None


class LatOccNeuralMotionPlanner(BaseNeuralMotionPlanner):
    OCC_COST_FACTOR = 200.0

    def __init__(self, n_input, n_output, pc_range, voxel_size, occ_loss_factor=1.0):
        super(LatOccNeuralMotionPlanner, self).__init__(
            n_input, n_output, pc_range, voxel_size
        )
        # an additional decoder for predicting future latent occupancy
        self.occ_decoder = Decoder(self.encoder.out_channels, self._out_channels)
        # rendering
        self.render_loss = DifferentiableRenderingLoss()
        self.occ_loss_factor = occ_loss_factor
        print("occ loss factor is", self.occ_loss_factor)

    def compute_cost_margins(self, batch):
        return self._compute_L2(batch)

    def compute_render_loss(self, batch, sigma):
        output_origins = batch["output_origins"]
        output_points = batch["output_points"]
        self._normalize(output_origins)
        self._normalize(output_points)
        render_loss = self.render_loss.apply(sigma, output_origins, output_points)
        return render_loss

    def forward(self, batch, mode):
        results = {}

        # voxelize input lidar sweeps
        I = self.prepare_input(batch)

        # extract backbone feature maps
        feat = self.encoder(I)

        # compute cost maps
        # imitation learning component
        il_cost = self.imitation_decoder(feat)

        # latent occupancy component
        sigma = F.relu(self.occ_decoder(feat), inplace=True)
        occ_cost = (1 - torch.exp(-sigma)) * self.OCC_COST_FACTOR

        # final cost map is a combination of two
        C = il_cost + occ_cost

        # clamp cost
        C = self.clamp_cost_maps(C)

        # evaluate the cost of every sampled trajectory
        CS = self.evaluate_samples(batch, C)

        if mode == "train":
            # evaluate the cost of the expert trajectory
            CG = self.evaluate_expert(batch, C)

            # compute cost margins (model-specific)
            CM = self.compute_cost_margins(batch)

            # construct the max-margin loss
            Lm, _ = ((F.relu(CG - CS + CM)).sum(dim=-1)).max(dim=-1)

            # return the margin loss
            results["margin_loss"] = Lm

            # second part of the loss: differentiable rendering loss
            Lr = self.compute_render_loss(batch, sigma)

            #
            results["render_loss"] = Lr

            #
            results["loss"] = Lm + Lr * self.occ_loss_factor

        else:
            results["il_cost"] = il_cost
            results["occ_cost"] = occ_cost
            results["cost"] = C
            results["best_plans"] = self.select_best_plans(batch, CS, 5)

        return results


class LatOccVFSupervisedNeuralMotionPlanner(BaseNeuralMotionPlanner):
    NVF_COST_FACTOR = 200.0

    def __init__(
        self, n_input, n_output, pc_range, voxel_size, nvf_loss_factor=1.0, dilate=False
    ):
        super(LatOccVFSupervisedNeuralMotionPlanner, self).__init__(
            n_input, n_output, pc_range, voxel_size
        )
        # decode into latent occupancy
        self.encoder = Encoder(
            self._in_channels, [2, 2, 3, 6, 5], [32, 64, 128, 256, 256]
        )
        self.occ_decoder = Decoder(self.encoder.out_channels, self._out_channels)
        # render latent occupancy into visible freespace and non-visible freespace
        # loss is still nvf classification
        self.nvf_loss_factor = nvf_loss_factor
        self.dilate = dilate
        print(
            f"LatOccVFSupervisedNeuralMotionPlanner: nvf loss factor is {nvf_loss_factor}"
        )

    def compute_cost_margins(self, batch):
        # cost margins are set via raycasted freespace estimates
        freespace = batch["fvf_maps"]

        # discretize sampled trajectories
        sampled_trajectories = batch["sampled_trajectories"]
        N = len(sampled_trajectories)
        ii = torch.arange(N)
        ti = torch.arange(self.n_output)
        Syi, Sxi = self._discretize(sampled_trajectories)

        # index observed future visible freespace with sampled trajectories
        # observed freespace is marked as -1
        label = freespace[ii[:, None, None], ti[None, None, :], Syi, Sxi]

        #
        nvf_cost = self.NVF_COST_FACTOR * (label != -1).float()

        #
        l2_cost = self._compute_L2(batch)

        return nvf_cost + l2_cost

    def compute_nvf_target(self, batch):
        # labels are set by raycasted freespace estimates
        freespace = batch["fvf_maps"]

        # positive: non visible freespace
        target = freespace != -1
        return target

    def perform_raymax(self, batch, sigma):
        output_origins = batch["output_origins"]
        output_points = batch["output_points"]
        self._normalize(output_origins)
        self._normalize(output_points)

        argmax_yy, argmax_xx = raymaxer.argmax(sigma, output_origins, output_points)
        # print(argmax_yy.shape, argmax_xx.shape, "shape after raymax")
        argmax_yy = argmax_yy.long()
        argmax_xx = argmax_xx.long()

        ii = torch.arange(len(output_origins))
        tt = torch.arange(self.n_output)

        nvf_logits = sigma[
            ii[:, None, None, None], tt[None, :, None, None], argmax_yy, argmax_xx
        ]

        return nvf_logits

    def forward(self, batch, mode):
        results = {}

        # voxelize input lidar sweeps
        I = self.prepare_input(batch)

        # extract backbone feature maps
        feat = self.encoder(I)

        # compute cost maps
        # imitation learning component
        il_cost = self.imitation_decoder(feat)

        # non-freespace component
        occ_logits = self.occ_decoder(feat)

        # perform raymax
        if mode == "train":
            nvf_logits = self.perform_raymax(batch, occ_logits)
            nvf_cost = torch.sigmoid(nvf_logits) * self.NVF_COST_FACTOR
            # final cost map is a combination of two
            C = il_cost + nvf_cost
        else:
            occ_cost = torch.sigmoid(occ_logits) * self.NVF_COST_FACTOR
            if self.dilate is True:
                print("occ cost shape before", occ_cost.shape)
                occ_cost = ndimage.morphology.grey_dilation(
                    occ_cost.cpu().numpy(), size=(1, 1, 10, 10)
                )
                occ_cost = torch.from_numpy(occ_cost).to("cuda")
                print("occ cost shape after", occ_cost.shape)
            C = il_cost + occ_cost

        # clamp cost
        C = self.clamp_cost_maps(C)

        # evaluate the cost of every sampled trajectory
        CS = self.evaluate_samples(batch, C)

        if mode == "train":
            # evaluate the cost of the expert trajectory
            CG = self.evaluate_expert(batch, C)

            # compute cost margins (model-specific)
            CM = self.compute_cost_margins(batch)

            # construct the max-margin loss
            Lm, _ = ((F.relu(CG - CS + CM)).sum(dim=-1)).max(dim=-1)

            # return the margin loss
            results["margin_loss"] = Lm

            # second part of the loss: visible freespace classification
            nvf_target = self.compute_nvf_target(batch).float()

            # binary cross entropy loss
            Lf = F.binary_cross_entropy_with_logits(nvf_logits, nvf_target)

            #
            results["nvf_loss"] = Lf

            #
            results["loss"] = Lm + Lf * self.nvf_loss_factor

        else:
            results["il_cost"] = il_cost
            results["occ_prob"] = torch.sigmoid(occ_logits)
            results["nvf_prob"] = torch.sigmoid(self.perform_raymax(batch, occ_logits))
            results["occ_cost"] = occ_cost
            results["cost"] = C
            results["best_plans"] = self.select_best_plans(batch, CS, 5)
            # results["best_plans"] = self.select_best_plans_given_gt(batch, 5)

        return results


class LatOccVFSupervisedL2CostMarginNeuralMotionPlanner(
    LatOccVFSupervisedNeuralMotionPlanner
):
    NVF_COST_FACTOR = 200.0

    def __init__(self, n_input, n_output, pc_range, voxel_size, nvf_loss_factor=1.0):
        super(LatOccVFSupervisedL2CostMarginNeuralMotionPlanner, self).__init__(
            n_input, n_output, pc_range, voxel_size, nvf_loss_factor
        )
        # decode into latent occupancy
        self.nvf_loss_factor = nvf_loss_factor
        print(
            f"LatOccVFSupervisedXLNeuralMotionPlanner: nvf loss factor is {nvf_loss_factor}"
        )

    def compute_cost_margins(self, batch):
        l2_cost = self._compute_L2(batch)
        return l2_cost


class LatOccVFSupervisedXLNeuralMotionPlanner(LatOccVFSupervisedNeuralMotionPlanner):
    NVF_COST_FACTOR = 200.0

    def __init__(self, n_input, n_output, pc_range, voxel_size, nvf_loss_factor=1.0):
        super(LatOccVFSupervisedXLNeuralMotionPlanner, self).__init__(
            n_input, n_output, pc_range, voxel_size, nvf_loss_factor
        )
        # decode into latent occupancy
        self.encoder = Encoder(
            self._in_channels, [4, 4, 5, 8, 7], [32, 64, 128, 256, 256]
        )
        self.nvf_loss_factor = nvf_loss_factor
        print(
            f"LatOccVFSupervisedXLNeuralMotionPlanner: nvf loss factor is {nvf_loss_factor}"
        )


class LatOccVFSupervisedXXLNeuralMotionPlanner(LatOccVFSupervisedNeuralMotionPlanner):
    NVF_COST_FACTOR = 200.0

    def __init__(self, n_input, n_output, pc_range, voxel_size, nvf_loss_factor=1.0):
        super(LatOccVFSupervisedXXLNeuralMotionPlanner, self).__init__(
            n_input, n_output, pc_range, voxel_size, nvf_loss_factor
        )
        # decode into latent occupancy
        self.encoder = Encoder(
            self._in_channels, [4, 4, 6, 12, 10], [32, 64, 128, 256, 256]
        )
        self.occ_decoder = Decoder(self.encoder.out_channels, self._out_channels)
        # render latent occupancy into visible freespace and non-visible freespace
        # loss is still nvf classification
        self.nvf_loss_factor = nvf_loss_factor
        print(
            f"LatOccVFSupervisedXXLNeuralMotionPlanner: nvf loss factor is {nvf_loss_factor}"
        )


class LatOccVFSupervisedXXXLNeuralMotionPlanner(LatOccVFSupervisedNeuralMotionPlanner):
    NVF_COST_FACTOR = 200.0

    def __init__(self, n_input, n_output, pc_range, voxel_size, nvf_loss_factor=1.0):
        super(LatOccVFSupervisedXXXLNeuralMotionPlanner, self).__init__(
            n_input, n_output, pc_range, voxel_size, nvf_loss_factor
        )
        # decode into latent occupancy
        self.encoder = Encoder(
            self._in_channels, [6, 6, 9, 18, 15], [32, 64, 128, 256, 256]
        )
        self.nvf_loss_factor = nvf_loss_factor
        print(
            f"LatOccVFSupervisedXXLNeuralMotionPlanner: nvf loss factor is {nvf_loss_factor}"
        )


class LatOccVFSupervisedLatOccNeuralMotionPlanner(
    LatOccVFSupervisedNeuralMotionPlanner
):
    NVF_COST_FACTOR = 200.0

    def __init__(self, n_input, n_output, pc_range, voxel_size, nvf_loss_factor=1.0):
        super(LatOccVFSupervisedLatOccNeuralMotionPlanner, self).__init__(
            n_input, n_output, pc_range, voxel_size, nvf_loss_factor
        )
        # decode into latent occupancy
        self.encoder = Encoder(
            self._in_channels, [2, 2, 3, 6, 5], [32, 64, 128, 256, 256]
        )
        self.occ_decoder = Decoder(self.encoder.out_channels, self._out_channels)
        # render latent occupancy into visible freespace and non-visible freespace
        # loss is still nvf classification
        self.nvf_loss_factor = nvf_loss_factor
        print(
            f"LatOccVFSupervisedLatOccNeuralMotionPlanner: nvf loss factor is {nvf_loss_factor}"
        )

    def forward(self, batch, mode):
        results = {}

        # voxelize input lidar sweeps
        I = self.prepare_input(batch)

        # extract backbone feature maps
        feat = self.encoder(I)

        # compute cost maps
        # imitation learning component
        il_cost = self.imitation_decoder(feat)

        # non-freespace component
        occ_logits = self.occ_decoder(feat)

        # perform raymax
        nvf_logits = self.perform_raymax(batch, occ_logits)
        # final cost map is a combination of residual cost map and occupancy cost map
        occ_cost = torch.sigmoid(occ_logits) * self.NVF_COST_FACTOR
        C = il_cost + occ_cost

        # clamp cost
        C = self.clamp_cost_maps(C)

        # evaluate the cost of every sampled trajectory
        CS = self.evaluate_samples(batch, C)

        if mode == "train":
            # evaluate the cost of the expert trajectory
            CG = self.evaluate_expert(batch, C)

            # compute cost margins (model-specific)
            CM = self.compute_cost_margins(batch)

            # construct the max-margin loss
            Lm, _ = ((F.relu(CG - CS + CM)).sum(dim=-1)).max(dim=-1)

            # return the margin loss
            results["margin_loss"] = Lm

            # second part of the loss: visible freespace classification
            nvf_target = self.compute_nvf_target(batch).float()

            # binary cross entropy loss
            Lf = F.binary_cross_entropy_with_logits(nvf_logits, nvf_target)

            #
            results["nvf_loss"] = Lf

            #
            results["loss"] = Lm + Lf * self.nvf_loss_factor

        else:
            results["il_cost"] = il_cost
            results["occ_prob"] = torch.sigmoid(occ_logits)
            results["nvf_prob"] = torch.sigmoid(self.perform_raymax(batch, occ_logits))
            results["occ_cost"] = occ_cost
            results["cost"] = C
            results["best_plans"] = self.select_best_plans(batch, CS, 5)
            # results["best_plans"] = self.select_best_plans_given_gt(batch, 5)

        return results


class LatOccVFSparseSupervisedNeuralMotionPlanner(
    LatOccVFSupervisedNeuralMotionPlanner
):
    NVF_COST_FACTOR = 200.0

    def __init__(self, n_input, n_output, pc_range, voxel_size, nvf_loss_factor=1.0):
        super(LatOccVFSparseSupervisedNeuralMotionPlanner, self).__init__(
            n_input, n_output, pc_range, voxel_size, nvf_loss_factor
        )
        # decode into latent occupancy
        self.encoder = Encoder(
            self._in_channels, [2, 2, 3, 6, 5], [32, 64, 128, 256, 256]
        )
        self.occ_decoder = Decoder(self.encoder.out_channels, self._out_channels)
        # render latent occupancy into visible freespace and non-visible freespace
        # loss is still nvf classification
        self.nvf_loss_factor = nvf_loss_factor
        print(
            f"LatOccVFSparseSupervisedNeuralMotionPlanner: nvf loss factor is {nvf_loss_factor}"
        )

    def sparsify(self, batch, loss):
        print("%" * 10, loss.shape)
        return loss

    def forward(self, batch, mode):
        results = {}

        # voxelize input lidar sweeps
        I = self.prepare_input(batch)

        # extract backbone feature maps
        feat = self.encoder(I)

        # compute cost maps
        # imitation learning component
        il_cost = self.imitation_decoder(feat)

        # non-freespace component
        occ_logits = self.occ_decoder(feat)

        # perform raymax
        nvf_logits = self.perform_raymax(batch, occ_logits)
        # final cost map is a combination of residual cost map and occupancy cost map
        occ_cost = torch.sigmoid(occ_logits) * self.NVF_COST_FACTOR
        C = il_cost + occ_cost

        # clamp cost
        C = self.clamp_cost_maps(C)

        # evaluate the cost of every sampled trajectory
        CS = self.evaluate_samples(batch, C)

        if mode == "train":
            # evaluate the cost of the expert trajectory
            CG = self.evaluate_expert(batch, C)

            # compute cost margins (model-specific)
            CM = self.compute_cost_margins(batch)

            # construct the max-margin loss
            Lm, _ = ((F.relu(CG - CS + CM)).sum(dim=-1)).max(dim=-1)

            # return the margin loss
            results["margin_loss"] = Lm

            # second part of the loss: visible freespace classification
            nvf_target = self.compute_nvf_target(batch).float()

            # binary cross entropy loss
            Lf = F.binary_cross_entropy_with_logits(nvf_logits, nvf_target)

            # sparsify the freespace loss
            Lf = self.sparsify(batch, Lf)

            #
            results["nvf_loss"] = Lf

            #
            results["loss"] = Lm + Lf * self.nvf_loss_factor

        else:
            results["il_cost"] = il_cost
            results["occ_prob"] = torch.sigmoid(occ_logits)
            results["nvf_prob"] = torch.sigmoid(self.perform_raymax(batch, occ_logits))
            results["occ_cost"] = occ_cost
            results["cost"] = C
            results["best_plans"] = self.select_best_plans(batch, CS, 5)
            # results["best_plans"] = self.select_best_plans_given_gt(batch, 5)

        return results


class ObjSupervisedRaymaxNeuralMotionPlanner(BaseNeuralMotionPlanner):
    OBJ_COST_FACTOR = 200.0

    def __init__(self, n_input, n_output, pc_range, voxel_size, obj_loss_factor=1.0):
        super(ObjSupervisedRaymaxNeuralMotionPlanner, self).__init__(
            n_input, n_output, pc_range, voxel_size
        )
        # decode into latent occupancy
        self.obj_decoder = Decoder(self.encoder.out_channels, self._out_channels)
        # obj uses boxes label
        # cost margin use box shadow
        self.obj_loss_factor = obj_loss_factor
        print(
            f"ObjSupervisedRaymaxNeuralMotionPlanner: obj loss factor is {obj_loss_factor}"
        )

    def compute_cost_margins(self, batch):
        # cost margins are set via raycasted freespace estimates
        # freespace = batch["fvf_maps"]
        obj_maps = batch["obj_shadows"]

        # discretize sampled trajectories
        sampled_trajectories = batch["sampled_trajectories"]
        N = len(sampled_trajectories)
        ii = torch.arange(N)
        ti = torch.arange(self.n_output)
        Syi, Sxi = self._discretize(sampled_trajectories)

        # index observed future visible freespace with sampled trajectories
        # observed freespace is marked as -1
        label = obj_maps[ii[:, None, None], ti[None, None, :], Syi, Sxi]

        #
        obj_cost = self.OBJ_COST_FACTOR * (label != -1).float()

        #
        l2_cost = self._compute_L2(batch)

        return obj_cost + l2_cost

    def compute_obj_target(self, batch):
        # labels are set by raycasted freespace estimates
        # freespace = batch["fvf_maps"]
        obj_maps = batch["obj_boxes"]

        # positive: non visible freespace
        # target = (freespace != -1)
        target = obj_maps == True
        return target

    def perform_raymax(self, batch, sigma):
        output_origins = batch["output_origins"]
        output_points = batch["output_points"]
        self._normalize(output_origins)
        self._normalize(output_points)

        argmax_yy, argmax_xx = raymaxer.argmax(sigma, output_origins, output_points)
        argmax_yy = argmax_yy.long()
        argmax_xx = argmax_xx.long()

        ii = torch.arange(len(output_origins))
        tt = torch.arange(self.n_output)

        obj_logits = sigma[
            ii[:, None, None, None], tt[None, :, None, None], argmax_yy, argmax_xx
        ]

        return obj_logits

    def forward(self, batch, mode):
        results = {}

        # voxelize input lidar sweeps
        I = self.prepare_input(batch)

        # extract backbone feature maps
        feat = self.encoder(I)

        # compute cost maps
        # imitation learning component
        il_cost = self.imitation_decoder(feat)

        # non-freespace component
        obj_logits = self.obj_decoder(feat)

        # perform raymax
        if mode == "train":
            obj_shadow_logits = self.perform_raymax(batch, obj_logits)
            obj_cost = torch.sigmoid(obj_shadow_logits) * self.OBJ_COST_FACTOR
        else:
            obj_cost = torch.sigmoid(obj_logits) * self.OBJ_COST_FACTOR

        #
        C = il_cost + obj_cost

        # clamp cost
        C = self.clamp_cost_maps(C)

        # evaluate the cost of every sampled trajectory
        CS = self.evaluate_samples(batch, C)

        if mode == "train":
            # evaluate the cost of the expert trajectory
            CG = self.evaluate_expert(batch, C)

            # compute cost margins (model-specific)
            CM = self.compute_cost_margins(batch)

            # construct the max-margin loss
            Lm, _ = ((F.relu(CG - CS + CM)).sum(dim=-1)).max(dim=-1)

            # return the margin loss
            results["margin_loss"] = Lm

            # second part of the loss: visible freespace classification
            obj_target = self.compute_obj_target(batch).float()

            # binary cross entropy loss
            Lf = F.binary_cross_entropy_with_logits(obj_logits, obj_target)

            #
            results["obj_loss"] = Lf

            #
            results["loss"] = Lm + Lf * self.obj_loss_factor

        else:
            results["il_cost"] = il_cost
            results["obj_cost"] = obj_cost
            results["cost"] = C
            results["best_plans"] = self.select_best_plans(batch, CS, 5)

        return results


class WarpModule(nn.Module):
    """
    taken from https://github.com/gengshan-y/VCN/blob/00c4befdbdf4e42050867996a6f686f52086e01a/models/VCN.py#L110
    taken from https://github.com/NVlabs/PWC-Net/blob/master/PyTorch/models/PWCNet.py
    """

    def __init__(self, W, H):
        super(WarpModule, self).__init__()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        # xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        # yy = yy.view(1,1,H,W).repeat(B,1,1,1)
        xx = xx.view(1, 1, H, W)
        yy = yy.view(1, 1, H, W)
        self.register_buffer("grid", torch.cat((xx, yy), 1).float())

    def forward(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        B, C, H, W = x.size()
        vgrid = self.grid + flo

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = nn.functional.grid_sample(x, vgrid)
        mask = ((vgrid[:, :, :, 0].abs() < 1) * (vgrid[:, :, :, 1].abs() < 1)) > 0
        return output * mask.unsqueeze(1).float(), mask


class LatOccFlowVFSupervisedNeuralMotionPlanner(LatOccVFSupervisedNeuralMotionPlanner):
    NVF_COST_FACTOR = 200.0

    def __init__(self, n_input, n_output, pc_range, voxel_size, nvf_loss_factor=1.0):
        super(LatOccFlowVFSupervisedNeuralMotionPlanner, self).__init__(
            n_input, n_output, pc_range, voxel_size
        )
        # decode into latent occupancy
        self.occ_decoder = Decoder(
            self.encoder.out_channels, 2 * (self.n_output - 1) + 1
        )
        self.occ_warper = WarpModule(self.n_width, self.n_length)
        # render latent occupancy into visible freespace and non-visible freespace
        # loss is still nvf classification
        self.nvf_loss_factor = nvf_loss_factor
        print(
            f"LatOccFlowVFSupervisedNeuralMotionPlanner: nvf loss factor is {nvf_loss_factor}"
        )

    def warp_occ(self, occ_initial, occ_flow):
        occ_logits_list = [occ_initial]
        for t in range(self.n_output - 1):
            flow = occ_flow[:, (t * 2) : (t * 2 + 2), :, :]
            occ_logits, _ = self.occ_warper(occ_initial, flow)
            occ_logits_list.append(occ_logits)
        return torch.cat(occ_logits_list, dim=1)

    def forward(self, batch, mode):
        results = {}

        # voxelize input lidar sweeps
        I = self.prepare_input(batch)

        # extract backbone feature maps
        feat = self.encoder(I)

        # compute cost maps
        # imitation learning component
        il_cost = self.imitation_decoder(feat)

        # non-freespace component
        occ = self.occ_decoder(feat)

        #
        occ_initial = occ[:, 0, None, :, :]
        occ_flow = occ[:, 1:, :, :]
        occ_logits = self.warp_occ(occ_initial, occ_flow)

        # perform raymax
        if mode == "train":
            nvf_logits = self.perform_raymax(batch, occ_logits)
            nvf_cost = torch.sigmoid(nvf_logits) * self.NVF_COST_FACTOR
            # final cost map is a combination of two
            C = il_cost + nvf_cost
        else:
            occ_cost = torch.sigmoid(occ_logits) * self.NVF_COST_FACTOR
            C = il_cost + occ_cost

        # clamp cost
        C = self.clamp_cost_maps(C)

        # evaluate the cost of every sampled trajectory
        CS = self.evaluate_samples(batch, C)

        if mode == "train":
            # evaluate the cost of the expert trajectory
            CG = self.evaluate_expert(batch, C)

            # compute cost margins (model-specific)
            CM = self.compute_cost_margins(batch)

            # construct the max-margin loss
            Lm, _ = ((F.relu(CG - CS + CM)).sum(dim=-1)).max(dim=-1)

            # return the margin loss
            results["margin_loss"] = Lm

            # second part of the loss: visible freespace classification
            nvf_target = self.compute_nvf_target(batch).float()

            # binary cross entropy loss
            Lf = F.binary_cross_entropy_with_logits(nvf_logits, nvf_target)

            #
            results["nvf_loss"] = Lf

            #
            results["loss"] = Lm + Lf * self.nvf_loss_factor

        else:
            results["il_cost"] = il_cost
            results["occ_cost"] = occ_cost
            results["cost"] = C
            results["best_plans"] = self.select_best_plans(batch, CS, 5)

        return results


class LatOccMultiFlowVFSupervisedNeuralMotionPlanner(
    LatOccVFSupervisedNeuralMotionPlanner
):
    NVF_COST_FACTOR = 200.0

    def __init__(
        self, n_input, n_output, pc_range, voxel_size, flow_mode=3, nvf_loss_factor=1.0
    ):
        super(LatOccMultiFlowVFSupervisedNeuralMotionPlanner, self).__init__(
            n_input, n_output, pc_range, voxel_size, nvf_loss_factor=nvf_loss_factor
        )
        # decode into latent occupancy: 1 (initial) + 3 (x, y, score) * flow_mode * time steps
        self.flow_mode = flow_mode
        self.occ_decoder = Decoder(
            self.encoder.out_channels, 1 + 3 * flow_mode * (self.n_output - 1)
        )
        self.occ_warper = WarpModule(self.n_width, self.n_length)
        # render latent occupancy into visible freespace and non-visible freespace
        # loss is still nvf classification
        self.nvf_loss_factor = nvf_loss_factor
        print(
            f"LatOccMultiFlowVFSupervisedNeuralMotionPlanner: nvf loss factor is {nvf_loss_factor}"
        )

    def warp_occ(self, occ_initial, occ_flows):
        K = self.flow_mode
        occ_logits_list = [occ_initial]
        for t in range(self.n_output - 1):
            occ_scores = F.softmax(occ_flows[:, t * K * 3 : t * K * 3 + K, :, :], dim=1)
            occ_fields = torch.zeros_like(occ_initial).repeat((1, K, 1, 1))
            for k in range(K):
                flow = occ_flows[:, t * K * 3 + K + 2 * k : t * K * 3 + K + 2 * k + 2]
                occ_field, _ = self.occ_warper(occ_logits_list[t], flow)
                occ_fields[:, k, :, :] += occ_scores[:, k, :, :] * occ_field[:, 0, :, :]
            occ_logits_list.append(occ_fields.sum(dim=1, keepdim=True))
        return torch.cat(occ_logits_list, dim=1)

    def forward(self, batch, mode):
        results = {}

        # voxelize input lidar sweeps
        I = self.prepare_input(batch)

        # extract backbone feature maps
        feat = self.encoder(I)

        # compute cost maps
        # imitation learning component
        il_cost = self.imitation_decoder(feat)

        # non-freespace component
        occ = self.occ_decoder(feat)

        #
        occ_initial = occ[:, 0, None, :, :]
        occ_flows = occ[:, 1:, :, :]
        occ_logits = self.warp_occ(occ_initial, occ_flows)

        # perform raymax
        if mode == "train":
            nvf_logits = self.perform_raymax(batch, occ_logits)
            nvf_cost = torch.sigmoid(nvf_logits) * self.NVF_COST_FACTOR
            # final cost map is a combination of two
            C = il_cost + nvf_cost
        else:
            occ_cost = torch.sigmoid(occ_logits) * self.NVF_COST_FACTOR
            C = il_cost + occ_cost

        # clamp cost
        C = self.clamp_cost_maps(C)

        # evaluate the cost of every sampled trajectory
        CS = self.evaluate_samples(batch, C)

        if mode == "train":
            # evaluate the cost of the expert trajectory
            CG = self.evaluate_expert(batch, C)

            # compute cost margins (model-specific)
            CM = self.compute_cost_margins(batch)

            # construct the max-margin loss
            Lm, _ = ((F.relu(CG - CS + CM)).sum(dim=-1)).max(dim=-1)

            # return the margin loss
            results["margin_loss"] = Lm

            # second part of the loss: visible freespace classification
            nvf_target = self.compute_nvf_target(batch).float()

            # binary cross entropy loss
            Lf = F.binary_cross_entropy_with_logits(nvf_logits, nvf_target)

            #
            results["nvf_loss"] = Lf

            #
            results["loss"] = Lm + Lf * self.nvf_loss_factor

        else:
            results["il_cost"] = il_cost
            results["occ_cost"] = occ_cost
            results["cost"] = C
            results["best_plans"] = self.select_best_plans(batch, CS, 5)

        return results
