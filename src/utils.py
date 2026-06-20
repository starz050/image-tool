"""
工具函数模块
"""

import cv2
import numpy as np
from pathlib import Path


def load_image(image_path):
    """
    加载图片
    
    Args:
        image_path: 图片路径
        
    Returns:
        numpy array: 图片数据 (BGR格式)
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")
    return image


def save_image(image, output_path):
    """
    保存图片
    
    Args:
        image: 图片数据 (numpy array)
        output_path: 输出路径
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def resize_image(image, width=None, height=None, inter=cv2.INTER_AREA):
    """
    调整图片大小
    
    Args:
        image: 输入图片
        width: 目标宽度
        height: 目标高度
        inter: 插值方法
        
    Returns:
        调整后的图片
    """
    (h, w) = image.shape[:2]
    
    if width is None and height is None:
        return image
    
    if width is None:
        ratio = height / float(h)
        width = int(w * ratio)
    
    if height is None:
        ratio = width / float(w)
        height = int(h * ratio)
    
    return cv2.resize(image, (width, height), interpolation=inter)


def convert_to_grayscale(image):
    """
    转换为灰度图
    
    Args:
        image: 输入图片 (BGR)
        
    Returns:
        灰度图
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_morphology(image, operation='close', kernel_size=5):
    """
    应用形态学操作
    
    Args:
        image: 输入图片
        operation: 操作类型 ('close', 'open', 'dilate', 'erode')
        kernel_size: 核大小
        
    Returns:
        处理后的图片
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    if operation == 'close':
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    elif operation == 'open':
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    elif operation == 'dilate':
        return cv2.dilate(image, kernel, iterations=1)
    elif operation == 'erode':
        return cv2.erode(image, kernel, iterations=1)
    else:
        raise ValueError(f"未知的操作: {operation}")


def detect_edges(image, low_threshold=50, high_threshold=150):
    """
    边缘检测 (Canny)
    
    Args:
        image: 输入图片
        low_threshold: 低阈值
        high_threshold: 高阈值
        
    Returns:
        边缘图
    """
    if len(image.shape) == 3:
        image = convert_to_grayscale(image)
    
    return cv2.Canny(image, low_threshold, high_threshold)


def find_contours(image):
    """
    查找轮廓
    
    Args:
        image: 输入图片
        
    Returns:
        轮廓列表
    """
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def get_image_info(image):
    """
    获取图片信息
    
    Args:
        image: 输入图片
        
    Returns:
        dict: 图片信息
    """
    h, w = image.shape[:2]
    channels = 1 if len(image.shape) == 2 else image.shape[2]
    
    return {
        'width': w,
        'height': h,
        'channels': channels,
        'size': image.nbytes,
    }
