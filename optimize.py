# optimize.py
# we use this script to check the implementation of renderer. 
import torch
import numpy as np
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes
from torch._C import import_ir_module_from_buffer
from data import nuScenesDataset, CollateFn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.cpp_extension import load

torch.random.manual_seed(0)
np.random.seed(0)

renderer = load("renderer", sources=[
    "lib/render/renderer.cpp", "lib/render/renderer.cu"], verbose=True)
raycaster = load("raycaster", sources=[
    "lib/raycast/raycaster.cpp", "lib/raycast/raycaster.cu"], verbose=True)
raymaxer = load("raymaxer", sources=[
    "lib/raymax/raymaxer.cpp", "lib/raymax/raymaxer.cu"], verbose=True)

nusc = NuScenes("v1.0-mini", "/data/nuscenes")

dataset_kwargs = {"n_input": 20, "n_samples": 100, "n_output": 7}
dataset = nuScenesDataset(nusc, "train", dataset_kwargs)

data_loader_kwargs = {"pin_memory": False, "shuffle": True, "batch_size": 1, "num_workers": 1}
data_loader = DataLoader(dataset, collate_fn=CollateFn, **data_loader_kwargs)

pc_range = [-40.0, -70.4, -2.0, 40.0, 70.4, 3.4]
voxel_size = 0.2
n_iter = 10000
output_grid = [7, 704, 400]

device = torch.device("cuda:0")
batch = next(iter(data_loader))

output_origins = batch["output_origins"].to(device)
output_points = batch["output_points"].to(device)

offset = torch.nn.parameter.Parameter(
    torch.Tensor(pc_range[:3])[None, None, :], requires_grad=False).to(device)
scaler = torch.nn.parameter.Parameter(
    torch.Tensor([voxel_size]*3)[None, None, :], requires_grad=False).to(device)

output_origins[:, :, :3] = (output_origins[:, :, :3] - offset) / scaler
output_points[:, :, :3] = (output_points[:, :, :3] - offset) / scaler

# what we would like
freespace = raycaster.raycast(output_origins, output_points, output_grid)
plt.imsave("sanity_checks/freespace.png", np.concatenate(freespace[0].detach().cpu().numpy(), axis=-1)[::-1])

sigma = torch.zeros((1, 7, 704, 400)).to(device)

# sigma = (freespace >= 0).float()
sigma = (freespace == 1).float()

# xi = (output_points[0, :, 0]).long()
# yi = (output_points[0, :, 1]).long()
# ti = (output_points[0, :, 3]).long()
# label = (output_points[0, :, 4]).int()
# xm = torch.logical_and(xi >= 0, xi < 400)
# ym = torch.logical_and(yi >= 0, yi < 704)
# tm = torch.logical_and(ti >= 0, ti < 7)
# m = torch.logical_and(torch.logical_and(xm, ym), label != 24)
# sigma[0, ti[m], yi[m], xi[m]] = 1

sigma.grad = torch.zeros_like(sigma)
optimizer = torch.optim.Adam([sigma], lr=5e-4)

# optimization loop
for i in range(0, n_iter):
    # losses, grad_sigma = renderer.render(sigma, output_origins, output_points)
    # pred_dist, gt_dist, grad_sigma = renderer.render(sigma, output_origins, output_points)

    argmax_yy, argmax_xx = raymaxer.argmax(sigma, output_origins, output_points)
    argmax_yy = argmax_yy.long()
    argmax_xx = argmax_xx.long()

    ii = torch.arange(sigma.shape[0])
    tt = torch.arange(sigma.shape[1])
    raymax_sigma = sigma[ii[:, None, None, None], tt[None, :, None, None], argmax_yy, argmax_xx]

    sigma = sigma.detach().cpu().numpy()
    raymax_sigma = raymax_sigma.detach().cpu().numpy()
    mix1 = np.concatenate(sigma.squeeze(), axis=-1) + np.concatenate(raymax_sigma.squeeze(), axis=-1)
    freespace = freespace.detach().cpu().numpy()
    mix2 = np.concatenate(freespace.squeeze(), axis=-1)
    img = np.concatenate((mix1, mix2), axis=0)
    plt.imshow(img[::-1])
    plt.show()

    import ipdb
    ipdb.set_trace()

    optimizer.zero_grad()
    sigma.grad = grad_sigma
    optimizer.step()
    sigma = F.relu(sigma, inplace=True)
    losses = torch.abs(pred_dist - gt_dist)
    mask = torch.logical_and(pred_dist >= 0, gt_dist >= 0)
    # mask = (losses >= 0)
    count = mask.sum()
    loss = losses[mask].sum() / count
    print("iter:", i, "loss:", loss.item())
    if (i+1) in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]:
        #
        gg = torch.zeros(output_grid)
        pg = torch.zeros(output_grid)
        for t in range(output_grid[0]):
            idx = output_points[0, :, 3] == t
            disp = output_points[0, idx, :3] - output_origins[0, t, :]
            dist = torch.sqrt(torch.sum(disp[:, :2] ** 2, dim=-1))
            vx = disp[:, 0] / dist
            vy = disp[:, 1] / dist
            
            # gg
            # it is possible for gt_dist to be bigger than dist 
            # when we are dealing with a ground return
            gx = (output_origins[0, t, 0] + gt_dist[0, idx] * vx).long()
            gy = (output_origins[0, t, 1] + gt_dist[0, idx] * vy).long()
            m2 = torch.logical_and(torch.logical_and(0 <= gx, gx < 400), 
                                   torch.logical_and(0 <= gy, gy < 704))
            gg[t, gy[m2], gx[m2]] = 1

            # pg 
            px = (output_origins[0, t, 0] + pred_dist[0, idx] * vx).long()
            py = (output_origins[0, t, 1] + pred_dist[0, idx] * vy).long()
            m3 = torch.logical_and(torch.logical_and(0 <= px, px < 400), 
                                   torch.logical_and(0 <= py, py < 704))
            pg[t, py[m3], px[m3]] = 1
    
        img = np.concatenate(sigma[0].detach().cpu().numpy(), axis=-1)[::-1]
        plt.imsave(f"sanity_checks/sigma_{i+1:06}.png", img)

        img = np.concatenate(gg.detach().cpu().numpy(), axis=-1)[::-1]
        plt.imsave(f"sanity_checks/gg_{i+1:06}.png", img)

        img = np.concatenate(pg.detach().cpu().numpy(), axis=-1)[::-1]
        plt.imsave(f"sanity_checks/pg_{i+1:06}.png", img)

