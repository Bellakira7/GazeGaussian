mkdir -p logs
# 随便建一个目录专门用来存放评估过程中微调产生的临时 checkpoint
mkdir -p eval_checkpoints

python evaluate_metrics.py \
    --log \
    --logdir ./logs \
    --img_dir /home/kong/dataset/ETH-XGaze/ETH-XGaze_test \
    --data_names eth_xgaze \
    --evaluation_type input_target_images \
    --checkpoint_path ./eval_checkpoints \
    --load_gazegaussian_checkpoint /home/kong/dataset/ETH-XGaze/gazegaussian_ckp.pth \
    --gpu_id 0