![Teaser](images/teaser.png)

# Differentiable Raycasting for Self-supervised Occupancy Forecasting
By Tarasha Khurana\*, Peiyun Hu\*, Achal Dave, Jason Ziglar, David Held, and Deva Ramanan
\* equal contribution

## Citing us
You can find our paper on [ECVA](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/1105_ECCV_2022_paper.php) and [Springer](https://link.springer.com/chapter/10.1007/978-3-031-19839-7_21). If you find our work useful, please consider citing:
```
@inproceedings{khurana2022differentiable,
  title={Differentiable Raycasting for Self-Supervised Occupancy Forecasting},
  author={Khurana, Tarasha and Hu, Peiyun and Dave, Achal and Ziglar, Jason and Held, David and Ramanan, Deva},
  booktitle={European Conference on Computer Vision},
  pages={353--369},
  year={2022},
  organization={Springer}
}
```

## Setup
- Download [nuScenes](https://www.nuscenes.org/nuscenes) dataset, including the [CANBus](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/can_bus/README.md) extension, as we will use the recorded vehicle state data for trajectory sampling. (Tip: the code assumes they are stored under `/data/nuScenes`.)
- Download [ONCE](https://once-for-auto-driving.github.io/download.html) dataset. (Tip: the code assumes they are stored under `/data/once`)
- Install packages and libraries (via `conda` if possible), including `torch`, `torchvision`, `tensorboard`, `cudatoolkit-11.1`, `pcl>=1.9`, `pybind11`, `eigen3`, `cmake>=3.10`, `scikit-image`, `nuscenes-devkit`. For your convenience, and `environment.yml` file has also been provided. (Tip: verify location of python binary with `which python`.)
- Compile code for Lidar point cloud ground segmentation under `lib/grndseg` using CMake.

## Preprocessing
- Run `preprocess.py` to generate ground segmentations
- Run `precast.py` to generate future visible freespace maps and a set of freespace contour points
- Run `rasterize.py` to generate BEV object occupancy maps and object "shadow" maps.

## Training
Refer to `train.py`.

## Testing
Refer to `test_nusc.py` and `test_once.py`.

## Acknowledgements
Code heavily adapted from @peiyunh's [repository](https://github.com/peiyunh/ff) (Safe Local Motion Planning at CVPR '21).
