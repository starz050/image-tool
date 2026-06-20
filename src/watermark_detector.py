"""
水印检测模块
"""

import cv2
import numpy as np
from . import utils


class WatermarkDetector:
    """水印检测器"""
    
    def __init__(self):
        """初始化水印检测器"""
        pass
    
    def detect_text_watermark(self, image, threshold=0.5):
        """
        检测文本水印
        
        Args:
            image: 输入图片
            threshold: 阈值
            
        Returns:
            dict: 检测结果
        """
        gray = utils.convert_to_grayscale(image)
        
        # 应用二值化
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # 查找轮廓
        contours = utils.find_contours(binary)
        
        watermarks = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:  # 过滤小的噪声
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h) if h > 0 else 0
            
            # 文本特征：宽高比在0.1-10之间
            if 0.1 < aspect_ratio < 10:
                watermarks.append({
                    'x': x,
                    'y': y,
                    'width': w,
                    'height': h,
                    'area': area,
                    'confidence': 0.7
                })
        
        return {
            'detected': len(watermarks) > 0,
            'count': len(watermarks),
            'regions': watermarks
        }
    
    def detect_logo_watermark(self, image, min_area=500):
        """
        检测logo水印
        
        Args:
            image: 输入图片
            min_area: 最小面积
            
        Returns:
            dict: 检测结果
        """
        gray = utils.convert_to_grayscale(image)
        
        # 应用高斯模糊和阈值化
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
        
        # 形态学操作
        binary = utils.apply_morphology(binary, 'close', 5)
        
        # 查找轮廓
        contours = utils.find_contours(binary)
        
        logos = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            logos.append({
                'x': x,
                'y': y,
                'width': w,
                'height': h,
                'area': area,
                'confidence': 0.8
            })
        
        return {
            'detected': len(logos) > 0,
            'count': len(logos),
            'regions': logos
        }
    
    def detect_border(self, image, edge_threshold=50):
        """
        检测边框
        
        Args:
            image: 输入图片
            edge_threshold: 边缘阈值
            
        Returns:
            dict: 检测结果
        """
        gray = utils.convert_to_grayscale(image)
        h, w = gray.shape
        
        # 检测四周边缘
        top_edge = gray[:edge_threshold, :]
        bottom_edge = gray[-edge_threshold:, :]
        left_edge = gray[:, :edge_threshold]
        right_edge = gray[:, -edge_threshold:]
        
        borders = {
            'top': {
                'detected': top_edge.mean() > 200,
                'intensity': top_edge.mean()
            },
            'bottom': {
                'detected': bottom_edge.mean() > 200,
                'intensity': bottom_edge.mean()
            },
            'left': {
                'detected': left_edge.mean() > 200,
                'intensity': left_edge.mean()
            },
            'right': {
                'detected': right_edge.mean() > 200,
                'intensity': right_edge.mean()
            }
        }
        
        return {
            'detected': any(b['detected'] for b in borders.values()),
            'borders': borders
        }
    
    def detect_all(self, image):
        """
        检测所有类型的水印
        
        Args:
            image: 输入图片
            
        Returns:
            dict: 完整检测结果
        """
        text_result = self.detect_text_watermark(image)
        logo_result = self.detect_logo_watermark(image)
        border_result = self.detect_border(image)
        
        return {
            'text': text_result,
            'logo': logo_result,
            'border': border_result,
            'has_watermark': (
                text_result['detected'] or 
                logo_result['detected'] or 
                border_result['detected']
            )
        }
