import os
import cv2
import numpy as np
import torch
import argparse

from dataloader.eth_xgaze import get_train_loader
from configs.gazegaussian_options import BaseOptions

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
    # 1. 设置保存 GT 图像的绝对路径
    gt_dir = "/home/kong/ylq/GazeGaussian-main/results/gt_images"
    os.makedirs(gt_dir, exist_ok=True)

    # 2. 测试受试者列表 (带有 .h5 后缀以适配 DataLoader 并用于命名)
    test_subjects = [
        "0012.h5", "0022.h5", "0025.h5", "0047.h5", "0054.h5", 
        "0074.h5", "0086.h5", "0094.h5", "0096.h5", "0097.h5", 
        "0110.h5", "0113.h5", "0115.h5", "0116.h5", "0119.h5"
    ]

    # 初始化配置
    base_options = BaseOptions()
    parser = auto_argparse_from_class(base_options) # 确保你的脚本里有这个函数定义
    opt = parser.parse_args()

    print(f"开始从 .h5 提取 GT 图像...")
    print(f"保存路径: {gt_dir}")

    for subject_id in test_subjects:
        print(f"正在处理受试者: {subject_id}")
        
        # 使用 DataLoader 提取数据保证预处理一致
        test_data_loader = get_train_loader(
            opt, 
            data_dir=opt.img_dir, 
            batch_size=1, 
            num_workers=0, 
            evaluate="landmark", 
            is_shuffle=False, 
            dataset_name=opt.dataset_name,
            subject=subject_id 
        )
        
        for i, data in enumerate(test_data_loader):
            # 命名格式：例如 0012.h5_0000.png
            img_name = f"{subject_id}_{str(i).zfill(4)}.png"
            
            # 提取 Ground Truth 图像张量
            gt_tensor = data['image'].squeeze() # 去掉 batch 维度
            
            # 反归一化：将 CHW 张量转换为 HWC 的 Numpy 数组
            gt_numpy = gt_tensor.cpu().numpy().transpose(1, 2, 0) 
            gt_numpy = (gt_numpy * 255).clip(0, 255).astype(np.uint8)
            
            # RGB 转 BGR 以适配 OpenCV 保存标准
            gt_bgr = cv2.cvtColor(gt_numpy, cv2.COLOR_RGB2BGR)
            
            # 保存图片
            save_path = os.path.join(gt_dir, img_name)
            cv2.imwrite(save_path, gt_bgr)

    print("=========================================")
    print(f"所有 GT 图像已成功保存至: {gt_dir}")

if __name__ == "__main__":
    main()