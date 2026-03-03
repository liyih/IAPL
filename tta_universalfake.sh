python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --master_port 29579 \
    main.py \
    --batchsize 32 \
    --evalbatchsize 32 \
    --dataset_path "/path/to/dataset" \
    --train_selected_subsets 'car' 'cat' 'chair' 'horse' \
    --test_selected_subsets 'crn' 'cyclegan' 'dalle' 'biggan' 'deepfake' 'gaugan' 'glide_50_27' 'glide_100_10' 'glide_100_27' 'guided' 'imle' 'ldm_100' 'ldm_200' 'ldm_200_cfg' 'progan' 'san' 'seeingdark' 'stargan' 'stylegan'\
    --lr 0.005 \
    --model_name tta\
    --epoch 1 \
    --lr_drop 10 \
    --gate True \
    --pretrained_model /path/to/pretrain_model_universalfake \
    --eval \
    --condition True \
    --tta True \
    --tta_steps 2 \
    --ois True