import os
import glob
import json
import numpy as np

# 1. 抓取目录下所有的 result_*.json 文件
json_files = glob.glob("/home/kong/ylq/GazeGaussian/logs/wo2dcnn_test_20260304_121843/result_*.json")

if not json_files:
    print("❌ 没有找到任何 result_*.json 文件，请确认你是否在正确的目录下运行此脚本！")
    exit()

print(f"✅ 成功读取到 {len(json_files)} 个人物的评估结果！正在汇总...\n")

# 2. 准备一个字典，用来收集所有的分数
metrics_collection = {
    "gaze": [],
    "head": [],
    "ssim": [],
    "psnr": [],
    "lpips": [],
    "l1": [],
    "sim": [] 
}

# 3. 遍历每个文件，把分数塞进对应的列表里
for f in json_files:
    with open(f, 'r') as file:
        data = json.load(file)
        # 遍历我们关心的指标，如果 json 里有，就存起来
        for key in metrics_collection.keys():
            if key in data:
                metrics_collection[key].append(data[key])

# 4. 计算平均分并打印出完美的论文表格格式
print("==========================================")
print("         🏆 最终 CVPR 论文成绩单 🏆       ")
print("==========================================")

for key, values in metrics_collection.items():
    if len(values) > 0:
        mean_val = np.mean(values)
        
        # 针对不同指标做精美的格式化输出
        if key == "sim":
            # 论文习惯把相似度乘以 100 变成百分制 (例如 0.677 -> 67.749)
            print(f"Identity Similarity ↑ : {mean_val * 100:.3f}")
        elif key == "gaze":
            print(f"Gaze Angular Error ↓  : {mean_val:.3f}")
        elif key == "head":
            print(f"Head Pose Error ↓     : {mean_val:.3f}")
        elif key == "psnr":
            print(f"PSNR ↑                : {mean_val:.3f}")
        elif key == "ssim":
            print(f"SSIM ↑                : {mean_val:.3f}")
        elif key == "lpips":
            print(f"LPIPS ↓               : {mean_val:.3f}")
        elif key == "l1":
            print(f"L1 Error ↓            : {mean_val:.4f}")

print("==========================================")

# import os
# import glob
# import re
# import numpy as np

# # 获取所有日志文件
# log_files = glob.glob("eval_log_*.txt")

# if not log_files:
#     print("❌ 没有找到 eval_log_*.txt 文件！")
#     exit()

# metrics = {"gaze": [], "head": [], "sim": [], "ssim": [], "psnr": [], "lpips": [], "l1": []}

# for file_path in log_files:
#     with open(file_path, "r") as f:
#         lines = f.readlines()
        
#     # 从后往前找，找到最后一次打印的 (Avg of 100) 的那行
#     for line in reversed(lines):
#         if "(Avg of 100)" in line or "(Avg of " in line:
#             # 使用正则表达式提取各个数值
#             # 格式例如: Gaze: 6.622 | Head: 2.128 | Sim: 0.677 | SSIM: 0.823 | ...
#             try:
#                 metrics["gaze"].append(float(re.search(r"Gaze:\s*([\d\.]+)", line).group(1)))
#                 metrics["head"].append(float(re.search(r"Head:\s*([\d\.]+)", line).group(1)))
#                 metrics["sim"].append(float(re.search(r"Sim:\s*([\-\d\.]+)", line).group(1)))
#                 metrics["ssim"].append(float(re.search(r"SSIM:\s*([\d\.]+)", line).group(1)))
#                 metrics["psnr"].append(float(re.search(r"PSNR:\s*([\d\.]+)", line).group(1)))
#                 metrics["lpips"].append(float(re.search(r"LPIPS:\s*([\d\.]+)", line).group(1)))
#                 metrics["l1"].append(float(re.search(r"L1:\s*([\d\.]+)", line).group(1)))
#                 break # 找到最后一行就跳出，去查下一个文件
#             except Exception as e:
#                 pass

# print("==========================================")
# print("         🏆 从日志抢救的 CVPR 成绩单 🏆     ")
# print("==========================================")

# for key, values in metrics.items():
#     if len(values) > 0:
#         mean_val = np.mean(values)
#         if key == "sim":
#             print(f"Identity Similarity ↑ : {mean_val * 100:.3f}")
#         elif key == "gaze":
#             print(f"Gaze Angular Error ↓  : {mean_val:.3f}")
#         elif key == "head":
#             print(f"Head Pose Error ↓     : {mean_val:.3f}")
#         elif key == "psnr":
#             print(f"PSNR ↑                : {mean_val:.3f}")
#         elif key == "ssim":
#             print(f"SSIM ↑                : {mean_val:.3f}")
#         elif key == "lpips":
#             print(f"LPIPS ↓               : {mean_val:.3f}")
#         elif key == "l1":
#             print(f"L1 Error ↓            : {mean_val:.4f}")
# print("==========================================")