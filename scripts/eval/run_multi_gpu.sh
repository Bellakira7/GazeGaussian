#!/bin/bash

# 1. 🚀 定义实验名称和自动生成时间戳
EXP_NAME="GazeGaussian_ETH_Test"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")  # 生成类似 20260302_082221 的格式
OUT_DIR="logs/${EXP_NAME}_${TIMESTAMP}"

# 创建这个专属的文件夹
mkdir -p $OUT_DIR
echo "📁 已创建本次实验的专属日志文件夹: $OUT_DIR"

SUBJECTS=("0012" "0022" "0025" "0047" "0054" "0074" "0086" "0094" "0096" "0097" "0110" "0113" "0115" "0116" "0119")
NUM_GPUS=8

for i in "${!SUBJECTS[@]}"; do
    SUBJECT=${SUBJECTS[$i]}
    GPU_ID=$((i % NUM_GPUS))
    
    echo "正在将 Subject $SUBJECT 分配到 GPU $GPU_ID ..."
    
    # 2. 🚀 修改点：新增 --out_dir 参数，并把 txt 存入 OUT_DIR
    CUDA_VISIBLE_DEVICES=$GPU_ID python evaluate_metrics.py \
        --subject $SUBJECT \
        --num_epochs 100 \
        --num_workers 0 \
        --img_dir /home/kong/dataset/ETH-XGaze/ETH-XGaze_test \
        --data_names eth_xgaze \
        --evaluation_type input_target_images \
        --checkpoint_path ./eval_checkpoints \
        --load_gazegaussian_checkpoint /home/kong/dataset/ETH-XGaze/gazegaussian_ckp.pth \
        --out_dir $OUT_DIR \
        > ${OUT_DIR}/eval_log_${SUBJECT}.txt 2>&1 &
        
    if [ $(( (i+1) % NUM_GPUS )) -eq 0 ]; then
        echo "⏳ GPU 已满编，等待当前批次完成..."
        wait
    fi
done

wait
echo "🎉 所有人物并行评估完毕！日志和 JSON 已保存在: $OUT_DIR"