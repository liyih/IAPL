python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --master_port 29576 \
    main.py \
    --batchsize 32 \
    --evalbatchsize 32 \
    --dataset_path "/path/to/dataset" \
    --train_selected_subsets 'SDv14' \
    --test_selected_subsets 'Chameleon'\
    --lr 0.005 \
    --model_name tta\
    --epoch 1 \
    --lr_drop 10 \
    --gate True \
    --condition True \
    --pretrained_model /path/to/pretrain_model_genimage \
    --eval \
    --dataset Chameleon_SD \
    --tta True \
    --tta_steps 2 \
    --ois True \
# (1 mean      ) acc: 75.09; ap: 64.69; racc: 64.81; facc: 88.76;