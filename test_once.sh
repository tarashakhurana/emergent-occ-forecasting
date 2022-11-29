python test_once.py --model-dir models/once_vf_supervised_nmp_10000_labeled_train_correctsamples \
    --test-split val --test-epoch 14 \
    --batch-size 16 --num-workers 16 \
    --sampled-trajectories data+curves \
    --sample-set labeled_train --compute-dense-fvf-loss --compute-raydist-loss
