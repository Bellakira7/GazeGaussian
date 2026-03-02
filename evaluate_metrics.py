import argparse
import os
import random
import cv2
import numpy as np
import torch
import scipy.io
import wandb

# --- 新增 GazeGaussian 的配置导入 ---
from configs.gazegaussian_options import BaseOptions
# ------------------------------------

from gaze_estimation.xgaze_baseline_resnet import gaze_network
from utils.metrics_utils import (evaluate_consistency,
                                 evaluate_personal_calibration,
                                 evaluate_input_target_images,
                                 evaluate_gaze_transfer)

def auto_argparse_from_class(cls_instance):
    """从配置类自动生成命令行参数"""
    parser = argparse.ArgumentParser(description="GazeGaussian Evaluation Options")
    
    # 获取类实例中已有的所有属性名
    existing_vars = vars(cls_instance)

    for attribute, value in existing_vars.items():
        # 跳过可能导致冲突或不需要的内部属性
        if attribute.startswith('__'):
            continue
            
        if isinstance(value, bool):
            parser.add_argument(f'--{attribute}', action='store_true' if not value else 'store_false')
        elif isinstance(value, list):
            # 自动判断列表元素的类型
            list_type = type(value[0]) if len(value) > 0 else str
            parser.add_argument(f'--{attribute}', type=list_type, nargs='+', default=value)
        else:
            parser.add_argument(f'--{attribute}', type=type(value) if value is not None else str, default=value)
    
    # --- 关键修改：只有当 BaseOptions 真的缺这几个评估专用参数时，才补上 ---
    if "data_names" not in existing_vars:
        parser.add_argument("--data_names", type=str, nargs='+', default=["eth_xgaze"])
    
    if "evaluation_type" not in existing_vars:
        parser.add_argument("--evaluation_type", type=str, default="input_target_images")

    if "gpu_id" not in existing_vars:
        parser.add_argument("--gpu_id", type=int, default=0)

    if "log" not in existing_vars:
        parser.add_argument("--log", action="store_true", help="Enable wandb logging")

    if "num_images" not in existing_vars:
        parser.add_argument("--num_images", type=int, nargs='+', default=[100])
        
    if "num_workers" not in existing_vars:
        parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--subject", type=str, default=None, help="指定单独评估的人物 ID")
    parser.add_argument("--out_dir", type=str, default=".", help="保存 JSON 结果的文件夹路径")
    return parser

def load_cams(opt):
    cam_matrix = {}
    cam_distortion = {}
    cam_translation = {}
    cam_rotation = {}

    for name in opt.data_names:
        cam_matrix[name] = []
        cam_distortion[name] = []
        cam_translation[name] = []
        cam_rotation[name] = []
    
    # 仅保留 ETH-XGaze 的相机加载逻辑（针对你的数据集）
    if "eth_xgaze" in opt.data_names:
        for cam_id in range(18):
            cam_file_name = "/home/kong/ylq/GazeGaussian-main/configs/dataset/eth_xgaze/cam/cam" + str(cam_id).zfill(2) + ".xml"
            fs = cv2.FileStorage(cam_file_name, cv2.FILE_STORAGE_READ)
            cam_matrix["eth_xgaze"].append(fs.getNode("Camera_Matrix").mat())
            cam_distortion["eth_xgaze"].append(fs.getNode("Distortion_Coefficients").mat())
            cam_translation["eth_xgaze"].append(fs.getNode("cam_translation"))
            cam_rotation["eth_xgaze"].append(fs.getNode("cam_rotation"))
            fs.release()

    return cam_matrix, cam_distortion, cam_translation, cam_rotation

def main():
    torch.manual_seed(45)  # cpu
    torch.cuda.manual_seed(55)  # gpu
    np.random.seed(65)  # numpy
    random.seed(75)  # random and transforms
    torch.backends.cudnn.deterministic = True  # cudnn
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    img_dim = 224

    # --- 1. 使用 GazeGaussian 方式解析参数 ---
    base_options = BaseOptions()
    parser = auto_argparse_from_class(base_options)
    opt = parser.parse_args()
    # -----------------------------------------

    device = torch.device(f"cuda:{opt.gpu_id}" if torch.cuda.is_available() else "cpu")

    if opt.log:
        wandb.init(project="gazegaussian_evaluation", config={"gpu_id": opt.gpu_id})
        wandb.config.update(opt)

    # --- 2. 加载 Gaze Error 评估模型 (ResNet) ---
    path = "configs/config_models/epoch_16_resnet_correct_ckpt.pth.tar"
    model = gaze_network().to(device)
    state_dict = torch.load(path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict=state_dict["model_state"])
    model.eval()
    print("Gaze ResNet Evaluator Loaded.")

    face_model_load = np.loadtxt("/home/kong/ylq/GazeGaussian-main/configs/dataset/eth_xgaze/face_model.txt")
    cam_matrix, cam_distortion, cam_translation, cam_rotation = load_cams(opt)
    
    # --- 3. 执行评估 (将 args 替换为 opt) ---
    if opt.evaluation_type == "input_target_images":
        evaluate_input_target_images(
            device, opt, model, cam_matrix, cam_distortion, 
            face_model_load, img_dim, opt.evaluation_type
        )
    # ... (如果需要保留其他评估模式，照此替换 args 为 opt) ...
    else:
        print("Wrong evaluation type")

if __name__ == "__main__":
    main()