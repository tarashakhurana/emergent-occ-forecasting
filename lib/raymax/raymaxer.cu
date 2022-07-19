#include <torch/extension.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <iostream>

#define MAX_D 1105 // 704 + 400 + 1

template <typename scalar_t>
__global__ void argmax_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> sigma,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> origins,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> points,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> argmax_yy,
    torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> argmax_xx) {

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
                count ++;
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
            last_d = _d;
            step ++;
        }

        // the total number of voxels visited should not exceed this number
        assert(count <= MAX_D);

        // WHEN THERE IS AN INTERSECTION BETWEEN THE RAY AND THE VOXEL GRID
        if (count > 0) {
            const auto v0 = path[0];
            float amax_y = v0.y;
            float amax_x = v0.x;
            float max_s = sigma[n][t][v0.y][v0.x];
            for (int i = 0; i < count; i ++) {
                const auto v = path[i];
                const auto s = sigma[n][t][v.y][v.x];
                if (max_s < s) {
                    max_s = s;
                    amax_y = v.y;
                    amax_x = v.x;
                }
                argmax_yy[n][t][v.y][v.x] = amax_y;
                argmax_xx[n][t][v.y][v.x] = amax_x;
            }
        }
    }
}

std::vector<torch::Tensor> argmax_cuda(
    torch::Tensor sigma,
    torch::Tensor origins,
    torch::Tensor points) {

    //
    auto argmax_yy = torch::zeros_like(sigma);
    auto argmax_xx = torch::zeros_like(sigma);

    //
    const auto N = points.size(0); // batch size
    const auto M = points.size(1); // num of rays
    const int threads = 1024;
    const dim3 blocks((M + threads - 1) / threads, N);

    AT_DISPATCH_FLOATING_TYPES(sigma.type(), "argmax_cuda", ([&] {
                argmax_cuda_kernel<scalar_t><<<blocks, threads>>>(
                    sigma.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                    origins.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                    points.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                    argmax_yy.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
                    argmax_xx.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>());
            }));

    // synchronize
    cudaDeviceSynchronize();

    return {argmax_yy, argmax_xx};
}
