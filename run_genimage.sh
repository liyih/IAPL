python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --master_port 29581\
    main.py \
    --batchsize 32 \
    --evalbatchsize 32 \
    --dataset_path "/path/to/dataset" \
    --train_selected_subsets 'SDv14'\
    --test_selected_subsets 'ADM' 'BigGAN' 'glide' 'Midjourney' 'stable_diffusion_v_1_4' 'stable_diffusion_v_1_5' 'VQDM' 'wukong'\
    --lr 0.00005 \
    --model_name genimage_sd14\
    --dataset GenImage \
    --epoch 5 \
    --lr_drop 10 \
    --gate True \
    --condition True\
    --smooth True\
