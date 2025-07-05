import os
import cv2
import numpy as np
from pathlib import Path

dataset = 'D:\Project\Engineering\database\LoveDA'
# Test,Train,Val
# Urban，Rural
# images_png,masks_png

def read_masks_from_folder(dataset_path, split='Train', area='Urban'):
    """
    读取指定文件夹中的mask文件
    
    Args:
        dataset_path (str): 数据集根目录路径
        split (str): 数据集分割 ('Train', 'Val', 'Test')
        area (str): 区域类型 ('Urban', 'Rural')
    
    Returns:
        list: mask文件路径列表
        dict: mask文件名到路径的映射
    """
    mask_folder = os.path.join(dataset_path, split, area, 'masks_png')
    
    if not os.path.exists(mask_folder):
        print(f"警告: 文件夹不存在 {mask_folder}")
        return [], {}
    
    # 获取所有mask文件
    mask_files = []
    mask_dict = {}
    
    # 支持的图像格式
    supported_formats = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
    
    for file in os.listdir(mask_folder):
        file_path = os.path.join(mask_folder, file)
        if os.path.isfile(file_path):
            # 检查文件扩展名
            _, ext = os.path.splitext(file)
            if ext.lower() in supported_formats:
                mask_files.append(file_path)
                mask_dict[file] = file_path
    
    print(f"找到 {len(mask_files)} 个mask文件在 {mask_folder}")
    return mask_files, mask_dict

def load_mask(mask_path):
    """
    加载单个mask文件
    
    Args:
        mask_path (str): mask文件路径
    
    Returns:
        numpy.ndarray: 加载的mask数组
    """
    try:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"无法加载mask文件: {mask_path}")
            return None
        return mask
    except Exception as e:
        print(f"加载mask文件时出错 {mask_path}: {e}")
        return None

def load_all_masks(mask_files):
    """
    批量加载所有mask文件
    
    Args:
        mask_files (list): mask文件路径列表
    
    Returns:
        dict: 文件名到mask数组的映射
    """
    masks = {}
    for mask_path in mask_files:
        filename = os.path.basename(mask_path)
        mask = load_mask(mask_path)
        if mask is not None:
            masks[filename] = mask
    
    print(f"成功加载 {len(masks)} 个mask文件")
    return masks

# 示例使用
if __name__ == "__main__":
    # 读取Train/Urban文件夹中的masks
    mask_files, mask_dict = read_masks_from_folder(dataset, 'Train', 'Urban')
    # 检查类别数目
    for i, item in enumerate(mask_dict.items()):
        if i >= 10:
            break
        print(f"文件名: {item[0]}, 路径: {item[1]}")
        img = cv2.imread(item[1], cv2.IMREAD_GRAYSCALE)
        print(np.unique(img))

