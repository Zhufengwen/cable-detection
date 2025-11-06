import os
# 第一步：设置环境变量（在导入任何库之前！）
os.environ['DISPLAY'] = ':0'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# 第二步：导入库
import streamlit as st
import numpy as np
from PIL import Image

# 第三步：安全地导入OpenCV
try:
    import cv2
    # 设置OpenCV为优化模式
    cv2.setUseOptimized(True)
    OPENCV_AVAILABLE = True
except Exception as e:
    OPENCV_AVAILABLE = False
    st.warning(f"OpenCV图形功能不可用，但基础功能正常")

# 第四步：导入YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception as e:
    YOLO_AVAILABLE = False
    st.error(f"YOLO加载失败: {e}")

# 应用界面
st.title("电缆检测系统")

if uploaded_file is not None:
    # 安全地处理图片 - 只用数据处理功能，不用显示功能
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    
    # 如果OpenCV可用，用它的处理功能
    if OPENCV_AVAILABLE:
        # 只使用数据处理函数，不调用任何显示函数
        # cv2.imdecode()  安全
        # cv2.imshow()    危险（会调用libGL）
        pass
