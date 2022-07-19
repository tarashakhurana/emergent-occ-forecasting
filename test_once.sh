# python test.py --model-dir models/occupancy_imitation \
#     --test-split val --test-epoch 9 \
#     --batch-size 4 --num-workers 4

# python test.py --model-dir models/vanilla_nmp \
#     --test-split val --test-epoch 7 \
#     --batch-size 32 --num-workers 16

# python test.py --model-dir models/vf_guided_nmp \
#     --test-split val --test-epoch 8 \
#     --batch-size 32 --num-workers 16

# python test.py --model-dir models/vf_guided_nmp_run2 \
#     --test-split val --test-epoch 7 \
#     --batch-size 32 --num-workers 16

# python test.py --model-dir models/obj_guided_nmp \
#     --test-split val --test-epoch 13 \
#     --batch-size 32 --num-workers 16

# python test.py --model-dir models/obj_shadow_guided_nmp \
#     --test-split val --test-epoch 13 \
#     --batch-size 32 --num-workers 16

# for epoch in 7 8;
# do
# python test.py --model-dir models/obj_guided_nmp \
#     --test-split val --test-epoch $epoch \
#     --batch-size 32 --num-workers 16
# done

# python test.py --model-dir models/vf_explicit_nmp \
#     --test-split val --test-epoch 13 \
#     --batch-size 32 --num-workers 16

# python test.py --model-dir models/vf_explicit_nmp_100 \
#     --test-split val --test-epoch 14 \
#     --batch-size 32 --num-workers 16

# python test.py --model-dir models/occ_explicit_nmp_100 \
#     --test-split val --test-epoch 14 \
#     --batch-size 32 --num-workers 16

# python test.py --model-dir models/obj_explicit_nmp_100 \
#     --test-split val --test-epoch 8 \
#     --batch-size 32 --num-workers 16

# python test.py --model-dir models/vf_supervised_nmp_100 \
#     --test-split val --test-epoch 10 \
#     --batch-size 32 --num-workers 16

# python test.py --model-dir models/obj_supervised_nmp_100 \
#     --test-split val --test-epoch 10 \
#     --batch-size 32 --num-workers 16

# python test.py --model-dir models/obj_shadow_supervised_nmp_10000 \
#     --test-split val --test-epoch 7 \
#     --batch-size 32 --num-workers 16

# for epoch in 6 8;
# do
# python test.py --model-dir models/obj_supervised_raymax_nmp_10000 \
#     --test-split val --test-epoch $epoch \
#     --batch-size 32 --num-workers 16
# done

# python test.py --model-dir models/obj_supervised_raymax_nmp_10000 \
#     --test-split val --test-epoch 6 \
#     --batch-size 32 --num-workers 16

# python test.py --model-dir models/vf_supervised_nmp_10000 \
#     --test-split val --test-epoch 9 \
#     --batch-size 32 --num-workers 16

# python test.py --model-dir models/obj_supervised_nmp_10000 \
#     --test-split val --test-epoch 7 \
#     --batch-size 32 --num-workers 16

# python test.py --model-dir models/obj_supervised_nmp_10000_run2 \
#     --test-split val --test-epoch 8 \
#     --batch-size 32 --num-workers 16

# python test.py --model-dir models/obj_supervised_nmp_10000_run3 \
#     --test-split val --test-epoch 6 \
#     --batch-size 32 --num-workers 16

# python test.py --model-dir models/lat_occ_nmp.no_stopper \
#     --test-split val --test-epoch 10 \
#     --batch-size 32 --num-workers 16

# python test.py --model-dir models/lat_occ_nmp \
#     --test-split val --test-epoch 13 \
#     --batch-size 32 --num-workers 16

# python test.py --model-dir models/lat_occ_nmp_100 \
#     --test-split val --test-epoch 14 \
#     --batch-size 32 --num-workers 16

# python test.py --model-dir models/lat_occ_nmp_100 \
#     --test-split val --test-epoch 14 \
#     --batch-size 32 --num-workers 16 \
#     --n-samples 10000

# python test.py --model-dir models/lat_occ_nmp_100 \
#     --test-split val --test-epoch 14 \
#     --batch-size 32 --num-workers 16 \
#     --n-samples 40000

# for epoch in $(seq 10 14)
# do
# python test.py --model-dir models/lat_occ_nmp_100 \
#     --test-split val --test-epoch $epoch \
#     --batch-size 32 --num-workers 16
# done

# python test.py --model-dir models/lat_occ_vf_supervised_nmp_10000 \
#     --test-split val --test-epoch 0 \
#     --batch-size 32 --num-workers 16

# python test.py --model-dir models/lat_occ_vf_supervised_nmp_10000_run2 \
#     --test-split val --test-epoch 6 \
#     --batch-size 32 --num-workers 16

# python test_once.py --model-dir models/once_lat_occ_vf_supervised_xl_nmp_10000_train_22k \
#     --test-split val --test-epoch 9 \
#     --batch-size 36 --num-workers 16

# 14 8.5k 12 23k

# for epoch in 6;
# do
# python test_once.py --model-dir models/once_lat_occ_vf_supervised_nmp_10000_labeled_train_correctsamples \
#     --test-split val --test-epoch $epoch \
#     --batch-size 32 --num-workers 16 \
#     --sampled-trajectories data+curves --suffix data+curves_trash \
#     --sample-set labeled_train --plot-on
# done

for epoch in 14;
do
python test_once.py --model-dir models/once_vf_supervised_nmp_10000_labeled_train_correctsamples \
    --test-split val --test-epoch $epoch \
    --batch-size 16 --num-workers 16 \
    --sampled-trajectories data+curves --suffix data+curves \
    --sample-set train_22k --plot-on
done

"""
python test_once.py --model-dir models/once_lat_occ_vf_supervised_nmp_10000_train_2k_unlabeled_correctsamples_ff \
    --test-split val --test-epoch 14 \
    --batch-size 64 --num-workers 16 \
    --sampled-trajectories data+curves --suffix data+curves \
    --sample-set labeled_train --compute-dense-fvf-loss

python test_once.py --model-dir models/once_lat_occ_vf_supervised_nmp_10000_train_4k_unlabeled_correctsamples_ff \
    --test-split val --test-epoch 14 \
    --batch-size 64 --num-workers 16 \
    --sampled-trajectories data+curves --suffix data+curves \
    --sample-set labeled_train --compute-dense-fvf-loss
"""

# for epoch in 11;
# do
# python test_once.py --model-dir models/once_vf_explicit_nmp_expt2_10000_labeled_train_correctsamples \
#     --test-split val --test-epoch $epoch \
#     --batch-size 32 --num-workers 16 \
#     --sampled-trajectories data+curves --suffix data+curves_trash_raydist \
#     --sample-set labeled_train --compute-raydist-loss
# done

# for epoch in 10 11 12 13 14;
# do
# python test_once.py --model-dir models/once_lat_occ_vf_supervised_nmp_0.1_labeled_train_correctsamples \
#     --test-split val --test-epoch $epoch \
#     --batch-size 32 --num-workers 16 \
#     --sampled-trajectories data+curves --suffix data+curves \
#     --sample-set labeled_train
# done

# for epoch in 4 5 6 7 8 9 10 11 12 13 14;
# do
# python test_once.py --model-dir models/once_lat_occ_vf_supervised_lat_occ_nmp_0.1_labeled_train_correctsamples \
#     --test-split val --test-epoch $epoch \
#     --batch-size 32 --num-workers 16 \
#     --sampled-trajectories data+curves --suffix data+curves \
#     --sample-set labeled_train
# done

# for epoch in 4 5 6 7 8 9 10 11 12 13 14;
# do
# python test_once.py --model-dir models/once_lat_occ_vf_supervised_nmp_1000_labeled_train_correctsamples \
#     --test-split val --test-epoch $epoch \
#     --batch-size 32 --num-workers 16 \
#     --sampled-trajectories data+curves --suffix data+curves \
#     --sample-set labeled_train
# done

# python test.py --model-dir models/lat_occ_vf_supervised_nmp_10000_allsweeps \
#     --test-split val --test-epoch 1 \
#     --batch-size 32 --num-workers 16

# python test.py --model-dir models/lat_occ_vf_supervised_nmp_10000_allsweeps \
#     --test-split val --test-epoch 6 \
#     --batch-size 32 --num-workers 16

# for epoch in 5 8;
# do
# python test.py --model-dir models/lat_occ_vf_supervised_nmp_10000_allsweeps_maxiter1835 \
#     --test-split val --test-epoch $epoch \
#     --batch-size 32 --num-workers 16
# done

# python test.py --model-dir models/lat_occ_flow_vf_supervised_nmp_10000 \
#     --test-split val --test-epoch 6 \
#     --batch-size 32 --num-workers 16

# python test.py --model-dir models/lat_occ_flow_vf_supervised_nmp_10000 \
#     --test-split val --test-epoch 9 \
#     --batch-size 32 --num-workers 16

# python test.py --model-dir models/lat_occ_multiflow_3_vf_supervised_nmp_10000 \
#     --test-split val --test-epoch 7 \
#     --batch-size 32 --num-workers 16
