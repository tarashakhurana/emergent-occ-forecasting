#include <torch/extension.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <iostream>

#define MAX_D 1105 // 704 + 400 + 1

template <typename scalar_t>
__global__ void render_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> sigma,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> origins,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> points,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> pred_dist,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> gt_dist,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> loss,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> grad_sigma) {

    // batch index
    const auto n = blockIdx.y;

    // ray index
    const auto c = blockIdx.x * blockDim.x + threadIdx.x;

    // num of rays
    const auto M = points.size(1);
    const auto T = sigma.size(1);

    // we allocated more threads than num_rays
    if (c < M) {
        // ray end point
        const auto t = points[n][c][3];

        // if t or label < 0, it is a padded point
        if (t < 0) return;

        // invalid points
        assert(t < T);

        // grid shape
        const int vysize = sigma.size(2);
        const int vxsize = sigma.size(3);
        // assert(vzsize + vysize + vxsize <= MAX_D);

        // origin
        const double xo = origins[n][t][0];
        const double yo = origins[n][t][1];

        // end point
        const double xe = points[n][c][0];
        const double ye = points[n][c][1];

        // locate the voxel where the origin resides
        const int vxo = int(xo);
        const int vyo = int(yo);

        //
        const int vxe = int(xe);
        const int vye = int(ye);

        // NOTE: new
        int vx = vxo;
        int vy = vyo;

        // origin to end
        const double rx = xe - xo;
        const double ry = ye - yo;
        double gt_d = sqrt(rx * rx + ry * ry);

        // directional vector
        const double dx = rx / gt_d;
        const double dy = ry / gt_d;

        // In which direction the voxel ids are incremented.
        const int stepX = (dx >= 0) ? 1 : -1;
        const int stepY = (dy >= 0) ? 1 : -1;

        // Distance along the ray to the next voxel border from the current position (tMaxX, tMaxY, tMaxZ).
        const double next_voxel_boundary_x = vx + (stepX < 0 ? 0 : 1);
        const double next_voxel_boundary_y = vy + (stepY < 0 ? 0 : 1);

        // tMaxX, tMaxY, tMaxZ -- distance until next intersection with voxel-border
        // the value of t at which the ray crosses the first vertical voxel boundary
        double tMaxX = (dx!=0) ? (next_voxel_boundary_x - xo)/dx : DBL_MAX; //
        double tMaxY = (dy!=0) ? (next_voxel_boundary_y - yo)/dy : DBL_MAX; //

        // tDeltaX, tDeltaY, tDeltaZ --
        // how far along the ray we must move for the horizontal component to equal the width of a voxel
        // the direction in which we traverse the grid
        // can only be FLT_MAX if we never go in that direction
        const double tDeltaX = (dx!=0) ? stepX/dx : DBL_MAX;
        const double tDeltaY = (dy!=0) ? stepY/dy : DBL_MAX;

        int2 path[MAX_D];
        double csd[MAX_D];  // cumulative sum of sigma times delta
        double p[MAX_D];  // alpha
        double d[MAX_D];
        double dt[MAX_D];

        // forward raymarching with voxel traversal
        int step = 0;  // total number of voxels traversed
        int count = 0;  // number of voxels traversed inside the voxel grid
        double last_d = 0.0;  // correct initialization

        // voxel traversal raycasting
        bool was_inside = false;
        while (true) {
            bool inside = (0 <= vx && vx < vxsize) && (0 <= vy && vy < vysize);
            if (inside) { // now inside
                was_inside = true;
                path[count] = make_int2(vx, vy);
            } else if (was_inside) { // was inside but no longer
                // we know we are not coming back so terminate
                break;
            } else if (last_d > gt_d) { // started and ended outside
                break;
            } 
            // _d represents the ray distance has traveled before escaping the current voxel cell
            double _d = 0.0;
            // voxel traversal
            if (tMaxX < tMaxY) {            
                _d = tMaxX;
                vx += stepX;
                tMaxX += tDeltaX;
            } else {
                _d = tMaxY;
                vy += stepY;
                tMaxY += tDeltaY;
            }
            if (inside) {
                // get sigma at the current voxel
                const int2 &v = path[count];  // use the recorded index
                const double _sigma = sigma[n][t][v.y][v.x];
                const double _delta = max(0.0, _d - last_d);  // THIS TURNS OUT IMPORTANT
                const double sd = _sigma * _delta;
                if (count == 0) { // the first voxel inside
                    csd[count] = sd;
                    p[count] = 1 - exp(-sd);
                } else {
                    csd[count] = csd[count-1] + sd;
                    p[count] = exp(-csd[count-1]) - exp(-csd[count]);
                }
                // record the traveled distance
                d[count] = _d;
                dt[count] = _delta;
                // count the number of voxels we have escaped
                count ++;
            }
            last_d = _d;
            step ++;
        }

        // the total number of voxels visited should not exceed this number
        assert(count <= MAX_D);

        // WHEN THERE IS AN INTERSECTION BETWEEN THE RAY AND THE VOXEL GRID
        if (count > 0) {
            // compute the expected ray distance
            double exp_d = 0.0;
            for (int i = 0; i < count; i ++) {
                exp_d += p[i] * d[i];
            }

            // add an imaginary sample at the end point should gt_d exceeds max_d
            double p_out = exp(-csd[count-1]);
            double max_d = d[count-1];
            // if we have reached the end within the grid, this should be false
            // this will not affect gradient, but it will make loss more informative
            if (gt_d > max_d)
                exp_d += (p_out * gt_d);

            // write the rendered ray distance (max_d)
            pred_dist[n][c] = exp_d;
            gt_dist[n][c] = gt_d;
            loss[n][c] = abs(exp_d - gt_d);

            /* backprop from where the ray ends to where the ray starts */
            double dd_dsigma[MAX_D];
            for (int i = count - 1; i >= 0; i --) {
                // NOTE: probably need to double check again
                if (i == count - 1)
                    dd_dsigma[i] = p_out * max_d;
                else
                    dd_dsigma[i] = dd_dsigma[i+1] - exp(-csd[i]) * (d[i+1] - d[i]);
            }

            for (int i = count - 1; i >= 0; i --)
                dd_dsigma[i] *= dt[i];

            if (gt_d > max_d)
                for (int i = count - 1; i >= 0; i --)
                    dd_dsigma[i] -= dt[i] * p_out * gt_d;

            double dl_dd = 0.0;
            if (exp_d > gt_d) {
                dl_dd = 1.0;
            } else if (exp_d < gt_d) {
                dl_dd = -1.0;
            }

            // apply chain rule
            for (int i = 0; i < count; i ++) {
                const int2 &v = path[i];
                // NOTE: potential race conditions when writing gradients
                grad_sigma[n][t][v.y][v.x] += dl_dd * dd_dsigma[i];
            }
        }
    }
}

/*
 * input shape
 *   sigma      : N x T x H x L x W
 *   origin   : N x T x 3
 *   points   : N x M x 4
 * output shape
 *   dist     : N x M
 *   loss     : N x M
 *   grad_sigma : N x T x H x L x W
 */
std::vector<torch::Tensor> render_cuda(
    torch::Tensor sigma,
    torch::Tensor origins,
    torch::Tensor points) {

    const auto N = points.size(0); // batch size
    const auto M = points.size(1); // num of rays

    const auto device = sigma.device();

    // perform rendering
    auto loss = -torch::ones({N, M}, device);
    auto pred_dist = -torch::ones({N, M}, device);
    auto gt_dist = -torch::ones({N, M}, device);
    // auto stopper = torch::zeros_like(sigma);
    auto grad_sigma = torch::zeros_like(sigma);

    const int threads = 1024;
    const dim3 blocks((M + threads - 1) / threads, N);

    // synchronize
    cudaDeviceSynchronize();

    // render
    AT_DISPATCH_FLOATING_TYPES(sigma.type(), "render_cuda", ([&] {
                render_cuda_kernel<scalar_t><<<blocks, threads>>>(
                    sigma.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                    origins.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                    points.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                    pred_dist.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                    gt_dist.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                    loss.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                    grad_sigma.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>());
            }));

    // synchronize
    cudaDeviceSynchronize();

    // return {pred_dist, gt_dist, grad_sigma};
    // return {loss, grad_sigma};
    return {pred_dist, gt_dist, loss, grad_sigma};
}
