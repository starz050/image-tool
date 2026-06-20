"""
图片处理核心模块
"""

import cv2
import numpy as np
from pathlib import Path
from . import utils
from .watermark_detector import WatermarkDetector


class ImageProcessor:
    """图片处理器"""
    
    def __init__(self):
        """初始化图片处理器"""
        self.detector = WatermarkDetector()
    
    def remove_watermark(self, input_path, output_path=None, method='inpaint'):
        """
        去除水印
        
        Args:
            input_path: 输入图片路径
            output_path: 输出图片路径 (可选)
            method: 去除方法 ('inpaint', 'remove_border', 'auto')
            
        Returns:
            numpy array: 处理后的图片
        """
        # 加载图片
        image = utils.load_image(input_path)
        
        # 检测水印
        detection = self.detector.detect_all(image)
        
        if method == 'auto':
            # 根据检测结果选择方法
            if detection['border']['detected']:
                result = self._remove_border(image)
            else:
                result = self._remove_by_inpainting(image, detection)
        elif method == 'inpaint':
            result = self._remove_by_inpainting(image, detection)
        elif method == 'remove_border':
            result = self._remove_border(image)
        else:
            raise ValueError(f"未知的方法: {method}")
        
        # 保存结果
        if output_path:
            utils.save_image(result, output_path)
        
        return result
    
    def _remove_border(self, image):
        """
        移除边框
        
        Args:
            image: 输入图片
            
        Returns:
            处理后的图片
        """
        gray = utils.convert_to_grayscale(image)
        h, w = gray.shape
        
        # 确定要移除的边界
        threshold = 200
        
        # 从上下左右查找内容区域
        top = 0
        for i in range(h):
            if gray[i, :].mean() < threshold:
                top = i
                break
        
        bottom = h
        for i in range(h - 1, -1, -1):
            if gray[i, :].mean() < threshold:
                bottom = i + 1
                break
        
        left = 0
        for j in range(w):
            if gray[:, j].mean() < threshold:
                left = j
                break
        
        right = w
        for j in range(w - 1, -1, -1):
            if gray[:, j].mean() < threshold:
                right = j + 1
                break
        
        # 裁剪图片
        result = image[top:bottom, left:right]
        
        return result
    
    def _remove_by_inpainting(self, image, detection):
        """
        使用图像修复去除水印
        
        Args:
            image: 输入图片
            detection: 检测结果
            
        Returns:
            处理后的图片
        """
        result = image.copy()
        
        # 创建掩码
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # 标记文本水印区域
        for region in detection['text']['regions']:
            x, y = region['x'], region['y']
            w, h = region['width'], region['height']
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
        
        # 标记logo区域
        for region in detection['logo']['regions']:
            x, y = region['x'], region['y']
            w, h = region['width'], region['height']
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
        
        # 如果有掩码，进行修复
        if mask.sum() > 0:
            result = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
        
        return result
    
    def crop_to_content(self, input_path, output_path=None, margin=0):
        """
        裁剪到内容
        
        Args:
            input_path: 输入图片路径
            output_path: 输出图片路径 (可选)
            margin: 边距 (像素)
            
        Returns:
            numpy array: 裁剪后的图片
        """
        image = utils.load_image(input_path)
        gray = utils.convert_to_grayscale(image)
        
        # 查找非零像素
        coords = cv2.findNonZero(cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)[1])
        
        if coords is None:
            return image
        
        x, y, w, h = cv2.boundingRect(coords)
        
        # 应用边距
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(image.shape[1] - x, w + 2 * margin)
        h = min(image.shape[0] - y, h + 2 * margin)
        
        result = image[y:y+h, x:x+w]
        
        if output_path:
            utils.save_image(result, output_path)
        
        return result
    
    def enhance_image(self, input_path, output_path=None, brightness=0, contrast=1.0):
        """
        增强图片
        
        Args:
            input_path: 输入图片路径
            output_path: 输出图片路径 (可选)
            brightness: 亮度调整 (-127 to 127)
            contrast: 对比度调整 (0.5 to 3.0)
            
        Returns:
            numpy array: 增强后的图片
        """
        image = utils.load_image(input_path)
        
        # 调整对比度和亮度
        result = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
        
        if output_path:
            utils.save_image(result, output_path)
        
        return result
    
    def get_image_info(self, input_path):
        """
        获取图片信息
        
        Args:
            input_path: 输入图片路径
            
        Returns:
            dict: 图片信息
        """
        image = utils.load_image(input_path)
        return utils.get_image_info(image)
