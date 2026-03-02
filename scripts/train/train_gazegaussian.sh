python train_gazegaussian.py \
--batch_size 1 \
--name 'gazegaussian' \
--img_dir '/home/kong/dataset/ETH-XGaze/ETH-XGaze' \
--num_epochs 20 \
--num_workers 2 \
--lr 0.0001 \
--clip_grad \
--load_meshhead_checkpoint /home/kong/ylq/GazeGaussian/work_dirs/meshhead_2026_02_28_14_57_36/checkpoints/meshhead_epoch_9.pth
# --load_gazegaussian_checkpoint ./checkpoint/gazegaussian_ckp.pth \