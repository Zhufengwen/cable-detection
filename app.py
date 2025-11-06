import streamlit as st
import numpy as np
from PIL import Image
import cv2
import onnxruntime as ort
import os

st.title("电缆缺陷检测系统")

@st.cache_resource
def load_model():
    try:
        # 找到ONNX文件
        onnx_files = [f for f in os.listdir('.') if f.endswith('.onnx')]
        if not onnx_files:
            return None
        return ort.InferenceSession(onnx_files[0])
    except:
        return None

model = load_model()

if model:
    st.success("模型加载成功")
    
    # 显示模型信息
    input_info = model.get_inputs()[0]
    st.write(f"模型期望输入: {input_info.name}, 形状: {input_info.shape}")
    
    uploaded_file = st.file_uploader("上传图片", type=['jpg', 'png', 'jpeg'])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="原图", use_column_width=True)
        
        if st.button("检测"):
            try:
                # 简单的预处理
                img_np = np.array(image)
                img_resized = cv2.resize(img_np, (640, 640))
                img_normalized = img_resized / 255.0
                img_chw = img_normalized.transpose(2, 0, 1)
                input_data = img_chw[np.newaxis, :, :, :].astype(np.float32)
                
                st.write(f"实际输入形状: {input_data.shape}")
                
                # 推理
                outputs = model.run(None, {input_info.name: input_data})
                st.success("推理成功完成！")
                
                # 显示输出信息
                for i, output in enumerate(outputs):
                    st.write(f"输出 {i}: {output.shape}")
                    
            except Exception as e:
                st.error(f"错误: {e}")
else:
    st.error("模型加载失败")
