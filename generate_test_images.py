import argparse
import random
import os
import numpy as np
import torch

from configs.gazegaussian_options import BaseOptions
from dataloader.eth_xgaze import get_train_loader, get_val_loader
from trainer.gazegaussian_trainer import GazeGaussianTrainer
from utils.recorder import GazeGaussianTrainRecorder

def auto_argparse_from_class(cls_instance):
    """从配置类自动生成命令行参数"""
    parser = argparse.ArgumentParser(description="GazeGaussian Inference Options")
    
    for attribute, value in vars(cls_instance).items():
        if isinstance(value, bool):
            parser.add_argument(f'--{attribute}', action='store_true' if not value else 'store_false',
                                help=f"Flag for {attribute}, default is {value}")
        elif isinstance(value, list):
            parser.add_argument(f'--{attribute}', type=type(value[0]), nargs='+', default=value,
                                help=f"List for {attribute}, default is {value}")
        else:
            parser.add_argument(f'--{attribute}', type=type(value), default=value,
                                help=f"Argument for {attribute}, default is {value}")

    return parser

def main():
    """主函数：生成重定向测试图片"""
    
    # 1. 固定随机种子，确保可复现性
    torch.manual_seed(2024)  # cpu
    torch.cuda.manual_seed(2024)  # gpu
    np.random.seed(2024)  # numpy
    random.seed(2024)  # random and transforms
    torch.backends.cudnn.deterministic = True  # cudnn
    torch.backends.cudnn.benchmark = True

    # 2. 解析参数
    base_options  = BaseOptions()
    parser = auto_argparse_from_class(base_options)
    opt = parser.parse_args()

    # 3. 实例化记录器与训练器（推理引擎）
    recorder = GazeGaussianTrainRecorder(opt)
    trainer = GazeGaussianTrainer(opt, recorder)

    # 4. 硬编码你的 15 个测试集受试者 ID (完美对应你的 ETH-XGaze_test 目录)
    test_subjects = [
        "0012.h5", "0022.h5", "0025.h5", "0047.h5", "0054.h5", 
        "0074.h5", "0086.h5", "0094.h5", "0096.h5", "0097.h5", 
        "0110.h5", "0113.h5", "0115.h5", "0116.h5", "0119.h5"
    ]

    print(f"准备开始生成，共包含 {len(test_subjects)} 个测试受试者。")

    # 5. 遍历测试集生成图片
    for subject in test_subjects:
        print(f"=========================================")
        print(f"正在处理受试者: subject{subject} ...")
        
        # 加载特定受试者的数据
        # 注意：这里要求 batch_size 必须为 1，且不能打乱顺序 (is_shuffle=False)
        test_data_loader = get_train_loader(
            opt, 
            data_dir=opt.img_dir, 
            batch_size=1, 
            num_workers=opt.num_workers, 
            evaluate="landmark", 
            is_shuffle=False, 
            dataset_name=opt.dataset_name,
            subject=subject 
        )
        
        # 获取该受试者的图片总数 (标准情况应该是 200 张)
        total_images = len(test_data_loader)
        print(f"受试者 {subject} 共有 {total_images} 张测试图像。")
        
        # 遍历所有图片进行单图微调与生成
        for i, data in enumerate(test_data_loader):
            print(f"--> 正在生成 第 {i+1}/{total_images} 张图片...")
            
            # 第一步：在线微调 (Test-time Fine-tuning)
            # 论文中提到推理时需要花约30秒针对单张图像优化特定特征
            # n_epochs 的数值可能需要根据原作者配置进行微调，这里默认设定为 100
            # trainer.train_single_image(test_data_loader, n_epochs=100, index=i) 
            
            # 第二步：生成并保存重定向图片
            # 图片会被保存到 recorder.visualize 指定的目录下 (通常是 logs/ 或 results/)
            # key 被设置为 "0012_0000" 的格式，方便后续计算指标时一一对应
            image_key = f"{subject}_{str(i).zfill(4)}"
            trainer.evaluate_single_image(test_data_loader, key=image_key, start_frame=i)
            
    print("=========================================")
    print("所有测试受试者的图片生成完毕！")

if __name__ == "__main__":
    main()