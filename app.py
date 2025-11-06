import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

st.title("电缆缺陷检测系统")

@st.cache_resource
def load_model():
    return YOLO('improvements.pt')

model = load_model()

def draw_boxes_pil(image, results):
    """使用PIL绘制检测框，完全避免OpenCV图形调用"""
    draw = ImageDraw.Draw(image)
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # 获取框坐标
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())
            
            # 绘制矩形框
            draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
            
            # 添加标签
            label = f"Class {cls} ({conf:.2f})"
            draw.text((x1, y1-10), label, fill='red')
    
    return image

uploaded_file = st.file_uploader("上传电缆图片", type=['jpg', 'jpeg', 'png'])

if uploaded_file and model:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="原图", use_container_width=True)
    
    if st.button("开始检测"):
        with st.spinner("检测中..."):
            try:
                results = model(np.array(image), conf=0.35, device='cpu')
                
                # 使用PIL绘制结果，完全避免OpenCV
                result_image = draw_boxes_pil(image.copy(), results)
                
                with col2:
                    st.image(result_image, caption="检测结果", use_container_width=True)
                    st.success(f"检测完成！共发现 {len(results[0].boxes)} 个缺陷")
                    
            except Exception as e:
                st.error(f"检测错误: {e}")
