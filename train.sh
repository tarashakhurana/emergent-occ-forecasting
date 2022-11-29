python train.py \
    --model-type lat_occ_vf_supervised_lat_occ \
    --model-dir models/nusc_lat_occ_vf_supervised_lat_occ_nmp_10000 \
    --nvf-loss-factor 10000.0 \
    --n-input 20 --n-output 7 \
    --batch-size 64 --num-workers 16 \
    --dataset nusc --data-root /data/nuScenes \
    --data-version v1.0-trainval

# can use --train-on-all-sweeps when training on the once dataset
