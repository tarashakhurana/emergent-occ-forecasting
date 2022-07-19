#!/bin/bash
# python train.py --model-dir models/occupancy_imitation \
#     --n-input 20 --n-output 7 \
#     --batch-size 16 --num-workers 16

# python train.py \
#     --model-type vanilla \
#     --model-dir models/vanilla_nmp \
#     --n-input 20 --n-output 7 \
#     --batch-size 16 --num-workers 16 \

# python train.py \
#     --model-type vf_guided \
#     --model-dir models/vf_guided_nmp \
#     --n-input 20 --n-output 7 \
#     --batch-size 16 --num-workers 16

# python train.py \
#     --model-type vf_guided \
#     --model-dir models/vf_guided_nmp_run2 \
#     --n-input 20 --n-output 7 \
#     --batch-size 16 --num-workers 16

# python train.py \
#     --model-type obj_guided \
#     --model-dir models/obj_guided_nmp \
#     --n-input 20 --n-output 7 \
#     --batch-size 16 --num-workers 16

# python train.py \
#     --model-type obj_shadow_guided \
#     --model-dir models/obj_shadow_guided_nmp \
#     --n-input 20 --n-output 7 \
#     --batch-size 16 --num-workers 16

# python train.py \
#     --model-type vf_explicit \
#     --model-dir models/vf_explicit_nmp \
#     --n-input 20 --n-output 7 \
#     --batch-size 12 --num-workers 16

# python train.py \
#     --model-type obj_explicit \
#     --model-dir models/obj_explicit_nmp_100 \
#     --obj-loss-factor 100.0 \
#     --n-input 20 --n-output 7 \
#     --batch-size 12 --num-workers 16

# python train.py \
#     --model-type vf_supervised \
#     --model-dir models/vf_supervised_nmp_100 \
#     --nvf-loss-factor 100.0 \
#     --n-input 20 --n-output 7 \
#     --batch-size 12 --num-workers 16

# python train.py \
#     --model-type vf_supervised \
#     --model-dir models/vf_supervised_nmp_10000 \
#     --nvf-loss-factor 10000.0 \
#     --n-input 20 --n-output 7 \
#     --batch-size 12 --num-workers 16

python train.py \
    --model-type lat_occ_vf_supervised_lat_occ \
    --model-dir models/nusc_lat_occ_vf_supervised_lat_occ_nmp_10000 \
    --nvf-loss-factor 10000.0 \
    --n-input 20 --n-output 7 \
    --batch-size 64 --num-workers 16 \
    --dataset nusc --data-root /data3/tkhurana/datasets/nuScenes \
    --data-version v1.0-trainval

# python train.py \
#     --model-type lat_occ_vf_supervised \
#     --model-dir models/lat_occ_vf_supervised_nmp_10000_run2 \
#     --nvf-loss-factor 10000.0 \
#     --n-input 20 --n-output 7 \
#     --batch-size 12 --num-workers 16

# python train.py \
#     --model-type lat_occ_vf_supervised \
#     --model-dir models/lat_occ_vf_supervised_nmp_10000_run3 \
#     --nvf-loss-factor 10000.0 \
#     --n-input 20 --n-output 7 \
#     --batch-size 12 --num-workers 16

# python train.py \
#     --model-type obj_shadow_guided \
#     --model-dir models/once_obj_shadow_guided_nmp_labeled_train_correctsamples \
#     --train-on-all-sweeps \
#     --nvf-loss-factor 10000.0 \
#     --n-input 5 --n-output 7 \
#     --batch-size 32 --num-workers 16 \
#     --num-epoch 15 --sampled-trajectories curves

# python train.py \
#     --model-type lat_occ_vf_sparse_supervised \
#     --model-dir models/once_lat_occ_vf_sparse_supervised_nmp_10000_labeled_train \
#     --train-on-all-sweeps \
#     --nvf-loss-factor 10000.0 \
#     --n-input 5 --n-output 7 \
#     --batch-size 64 --num-workers 16 \
#     --num-epoch 15

# python train.py \
#     --data-version v1.0-trainval \
#     --model-type vf_supervised \
#     --model-dir models/nusc_vf_supervised_nmp_10000_ff \
#     --nvf-loss-factor 10000 \
#     --n-input 20 --n-output 7 \
#     --batch-size 64 --num-workers 16 \
#     --num-epoch 15

# python train.py \
#     --data-version v1.0-trainval \
#     --model-type lat_occ_vf_supervised \
#     --model-dir models/nusc_lat_occ_vf_supervised_nmp_10000_ff \
#     --nvf-loss-factor 10000 \
#     --n-input 20 --n-output 7 \
#     --batch-size 64 --num-workers 16 \
#     --num-epoch 15

# have removed train on all sweeps from the above command

# python train.py \
#     --dataset once \
#     --data-root /data3/tkhurana/datasets/once/ \
#     --data-version labeled_train \
#     --model-type lat_occ_vf_supervised \
#     --model-dir models/once_lat_occ_vf_supervised_nmp_10000_labeled_train_correctsamples_ff \
#     --nvf-loss-factor 10000 \
#     --n-input 5 --n-output 7 \
#     --train-on-all-sweeps \
#     --batch-size 64 --num-workers 16 \
#     --num-epoch 15

# python train.py \
#     --model-type lat_occ_vf_supervised \
#     --model-dir models/once_lat_occ_vf_supervised_nmp_10_labeled_train_correctsamples \
#     --train-on-all-sweeps \
#     --nvf-loss-factor 10.0 \
#     --n-input 5 --n-output 7 \
#     --batch-size 64 --num-workers 16 \
#     --num-epoch 15

# python train.py \
#     --model-type lat_occ_vf_supervised \
#     --model-dir models/once_lat_occ_vf_supervised_nmp_100_labeled_train_correctsamples \
#     --train-on-all-sweeps \
#     --nvf-loss-factor 100.0 \
#     --n-input 5 --n-output 7 \
#     --batch-size 64 --num-workers 16 \
#     --num-epoch 15

# python train.py \
#     --model-type lat_occ_vf_supervised \
    # --model-dir models/once_lat_occ_vf_supervised_nmp_1000_labeled_train_correctsamples \
    # --train-on-all-sweeps \
    # --nvf-loss-factor 1000.0 \
    # --n-input 5 --n-output 7 \
    # --batch-size 64 --num-workers 16 \
    # --num-epoch 15

# python train.py \
#     --model-type vf_explicit \
#     --model-dir models/once_vf_explicit_nmp_expt2_10000_labeled_train_correctsamples \
#     --train-on-all-sweeps \
#     --nvf-loss-factor 10000.0 \
#     --n-input 5 --n-output 7 \
#     --batch-size 64 --num-workers 16 \
#     --num-epoch 15

# python train.py \
#     --model-type vf_supervised \
#     --model-dir models/once_vf_supervised_nmp_10000_labeled_train_correctsamples \
#     --train-on-all-sweeps \
#     --nvf-loss-factor 10000.0 \
#     --n-input 5 --n-output 7 \
#     --batch-size 64 --num-workers 16 \
#     --num-epoch 15

# python train.py \
#     --model-type lat_occ_vf_supervised_l2_costmargin \
#     --model-dir models/once_lat_occ_vf_supervised_l2_costmargin_nmp_10000_labeled_train_correctsamples \
#     --train-on-all-sweeps \
#     --nvf-loss-factor 10000.0 \
#     --n-input 5 --n-output 7 \
#     --batch-size 64 --num-workers 16 \
#     --num-epoch 15

# python train.py \
#     --model-type lat_occ_vf_supervised \
#     --model-dir models/lat_occ_vf_supervised_nmp_10000_allsweeps_maxiter1835 \
#     --nvf-loss-factor 10000.0 \
#     --train-on-all-sweeps \
#     --max-iters-per-epoch 1835 \
#     --n-input 20 --n-output 7 \
#     --batch-size 12 --num-workers 16

# python train.py \
#     --model-type lat_occ_flow_vf_supervised \
#     --model-dir models/lat_occ_flow_vf_supervised_nmp_10000 \
#     --nvf-loss-factor 10000.0 \
#     --n-input 20 --n-output 7 \
#     --batch-size 12 --num-workers 16

# python train.py \
#     --model-type lat_occ_flow_vf_supervised \
#     --model-dir models/lat_occ_flow_vf_supervised_nmp_10000 \
#     --nvf-loss-factor 10000.0 \
#     --n-input 20 --n-output 7 \
#     --batch-size 12 --num-workers 16

# python train.py \
#     --model-type lat_occ_multiflow_vf_supervised \
#     --model-dir models/lat_occ_multiflow_3_vf_supervised_nmp_10000 \
#     --flow-mode 3 \
#     --nvf-loss-factor 10000.0 \
#     --n-input 20 --n-output 7 \
#     --batch-size 12 --num-workers 16

# python train.py \
#     --model-type obj_supervised \
#     --model-dir models/obj_supervised_nmp_100 \
#     --obj-loss-factor 100.0 \
#     --n-input 20 --n-output 7 \
#     --batch-size 12 --num-workers 16

# python train.py \
#     --model-type obj_supervised \
#     --model-dir models/obj_supervised_nmp_10000 \
#     --obj-loss-factor 10000.0 \
#     --n-input 20 --n-output 7 \
#     --batch-size 12 --num-workers 16

# python train.py \
#     --model-type obj_supervised \
#     --model-dir models/obj_supervised_nmp_10000_run2 \
#     --obj-loss-factor 10000.0 \
#     --n-input 20 --n-output 7 \
#     --batch-size 12 --num-workers 16

# python train.py \
#     --model-type obj_supervised \
#     --model-dir models/obj_supervised_nmp_10000_run3 \
#     --obj-loss-factor 10000.0 \
#     --n-input 20 --n-output 7 \
#     --batch-size 12 --num-workers 16

# python train.py \
#     --model-type obj_shadow_supervised \
#     --model-dir models/obj_shadow_supervised_nmp_10000 \
#     --obj-loss-factor 10000.0 \
#     --n-input 20 --n-output 7 \
#     --batch-size 12 --num-workers 16

# python train.py \
#     --model-type obj_supervised_raymax \
#     --model-dir models/obj_supervised_raymax_nmp_10000 \
#     --obj-loss-factor 10000.0 \
#     --n-input 20 --n-output 7 \
#     --batch-size 12 --num-workers 16

# python train.py \
#     --model-type vf_explicit \
#     --model-dir models/vf_explicit_nmp_100 \
#     --nvf-loss-factor 100.0 \
#     --n-input 20 --n-output 7 \
#     --batch-size 12 --num-workers 16

# python train.py \
#     --model-type lat_occ \
#     --model-dir models/lat_occ_nmp \
#     --render-loss-factor 1.0 \
#     --n-input 20 --n-output 7 \
#     --batch-size 8 --num-workers 16

# python train.py \
#     --model-type lat_occ \
#     --model-dir models/lat_occ_nmp_100 \
#     --occ-loss-factor 100.0 \
#     --n-input 20 --n-output 7 \
#     --batch-size 8 --num-workers 16

# python train.py \
#     --model-type lat_occ \
#     --model-dir models/lat_occ_nmp_10000 \
#     --occ-loss-factor 10000.0 \
#     --n-input 20 --n-output 7 \
#     --batch-size 8 --num-workers 16

# python train.py \
#     --model-type occ_explicit \
#     --model-dir models/occ_explicit_nmp_100 \
#     --n-input 20 --n-output 7 \
#     --occ-loss-factor 100.0 \
#     --batch-size 12 --num-workers 16
