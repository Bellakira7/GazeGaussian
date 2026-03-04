#!/bin/bash

# 1. 🚀 定义实验名称和自动生成时间戳
EXP_NAME="wo2dcnn_test"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")  # 生成类似 20260302_082221 的格式
OUT_DIR="logs/${EXP_NAME}_${TIMESTAMP}"

# 创建这个专属的文件夹
mkdir -p $OUT_DIR
echo "📁 已创建本次实验的专属日志文件夹: $OUT_DIR"

SUBJECTS=("0012" "0022" "0025" "0047" "0054" "0074" "0086" "0094" "0096" "0097" "0110" "0113" "0115" "0116" "0119")

# ==========================================
# 🌟 修改点 1：定义你要使用的具体 GPU 编号集合
# ==========================================
GPUS=(1 2 3 4 5 6 7) # 在括号里填入你想用的显卡编号，空格隔开。比如用三张卡就写 (1 4 7)
NUM_GPUS=${#GPUS[@]} # 自动计算数组长度，也就是你用了几张卡

for i in "${!SUBJECTS[@]}"; do
    SUBJECT=${SUBJECTS[$i]}
    
    # ==========================================
    # 🌟 修改点 2：把取余的结果当做“索引”，去数组里取真实的 GPU 编号
    # ==========================================
    GPU_INDEX=$((i % NUM_GPUS))
    GPU_ID=${GPUS[$GPU_INDEX]} 
    
    echo "正在将 Subject $SUBJECT 分配到 GPU $GPU_ID ..."
    
    CUDA_VISIBLE_DEVICES=$GPU_ID python evaluate_metrics.py \
        --subject $SUBJECT \
        --num_epochs 100 \
        --num_workers 0 \
        --num_images 10 \
        --img_dir /home/kong/dataset/ETH-XGaze/ETH-XGaze_test \
        --data_names eth_xgaze \
        --evaluation_type input_target_images \
        --checkpoint_path ./eval_checkpoints \
        --load_gazegaussian_checkpoint /home/kong/ylq/GazeGaussian/work_dirs/gazegaussian_2026_03_03_13_17_58/checkpoints/gazegaussian_epoch_1.pth \
        --out_dir $OUT_DIR \
        > ${OUT_DIR}/eval_log_${SUBJECT}.txt 2>&1 &
        
    # 并发控制逻辑无需修改，它会根据 NUM_GPUS 自动适应
    if [ $(( (i+1) % NUM_GPUS )) -eq 0 ]; then
        echo "⏳ 指定的 ${NUM_GPUS} 张 GPU 已满编，等待当前批次完成..."
        wait
    fi
done

wait
echo "🎉 所有人物并行评估完毕！日志和 JSON 已保存在: $OUT_DIR"