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

python test_nusc.py --model-dir models/nusc_lat_occ_vf_supervised_lat_occ_nmp_10000/ \
    --test-split val --test-epoch 14 \
    --batch-size 64 --num-workers 16

# for epoch in 6 7 8 9 10 11:
# do
# python test.py --model-dir models/nusc_lat_occ_vf_supervised_nmp_10000_train_8k_correctsamples_ff/ \
#     --test-split val --test-epoch $epoch \
#     --batch-size 32 --num-workers 16 \
#     --compute-dense-fvf-loss
# done

# for epoch in 6 7 8 9 10 11:
# do
# python test.py --model-dir models/nusc_lat_occ_vf_supervised_nmp_10000_train_16k_correctsamples_ff/ \
#     --test-split val --test-epoch $epoch \
#     --batch-size 32 --num-workers 16 \
#     --compute-dense-fvf-loss
# done

# python test.py --model-dir models/lat_occ_vf_supervised_nmp_10000_run2 \
#     --test-split val --test-epoch 9 \
#     --batch-size 32 --num-workers 16

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
