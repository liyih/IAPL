python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --master_port 29580 \
    main.py \
    --batchsize 16 \
    --evalbatchsize 32 \
    --dataset_path "/path/to/dataset" \
    --train_selected_subsets 'car' 'cat' 'chair' 'horse' \
    --test_selected_subsets 'crn' 'cyclegan' 'dalle' 'biggan' 'deepfake' 'gaugan' 'glide_50_27' 'glide_100_10' 'glide_100_27' 'guided' 'imle' 'ldm_100' 'ldm_200' 'ldm_200_cfg' 'progan' 'san' 'seeingdark' 'stargan' 'stylegan'\
    --lr 0.00005 \
    --model_name universalfake_progan\
    --epoch 5 \
    --lr_drop 10 \
    --gate True \
    --condition True \
    --smooth True \
