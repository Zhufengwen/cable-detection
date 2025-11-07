import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import torch
import torchvision.transforms as transforms
import os

st.set_page_config(
    page_title="电缆缺陷检测系统",
    layout="wide"
)

st.title("电缆缺陷检测系统")
st.markdown("上传电缆图片，AI自动检测各种缺陷类型")

@st.cache_resource
def load_model():
    """直接加载PyTorch模型，绕过YOLO库"""
    try:
        model_files = [f for f in os.listdir('.') if f.endswith('.pt')]
        if not model_files:
            st.error("未找到模型文件")
            return None
        
        model_path = model_files[0]
        st.info(f"加载模型: {model_path}")
        
        # 直接使用torch加载，避免YOLO库的OpenCV依赖
        model = torch.load(model_path, map_location='cpu')
        model.eval()
        
        st.success("模型加载成功!")
        return model
        
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        # 如果直接加载失败，尝试其他方案
        return try_alternative_loading()

def try_alternative_loading():
    """备选加载方案"""
    try:
        # 尝试使用torch.hub加载
        model = torch.hub.load('.', 'custom', path='improvements.pt', source='local')
        return model
    except:
        return None

# 类别映射
CLASS_NAMES = {
    0: "断裂股线", 1: "焊接股线", 2: "弯曲股线", 3: "长划痕",
    4: "压碎", 5: "间隔股线", 6: "沉积物", 7: "断裂", 8: "雷击损坏"
}

def simple_detection(image, model, conf_threshold=0.25):
    """简化的检测函数"""
    try:
        # 图像预处理
        transform = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
        ])
        
        input_tensor = transform(image).unsqueeze(0)
        
        # 推理
        with torch.no_grad():
            predictions = model(input_tensor)
        
        return process_predictions(predictions, image.size)
        
    except Exception as e:
        st.error(f"检测错误: {e}")
        return []

def process_predictions(predictions, original_size):
    """处理预测结果"""
    # 这里需要根据你的模型输出格式来解析
    # 这是一个通用模板，你需要根据实际模型调整
    detections = []
    
    # 假设predictions包含[boxes, scores, classes]
    if hasattr(predictions, 'xyxy'):
        # YOLO格式
        boxes = predictions.xyxy[0].cpu().numpy()
        for box in boxes:
            if len(box) >= 6:  # x1, y1, x2, y2, conf, class
                x1, y1, x2, y2, conf, cls = box[:6]
                if conf > 0.25:
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(conf),
                        'class_id': int(cls),
                        'class_name': CLASS_NAMES.get(int(cls), f"类别{int(cls)}")
                    })
    
    return detections

def draw_detections(image, detections):
    """绘制检测框"""
    draw = ImageDraw.Draw(image)
    
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        confidence = det['confidence']
        class_name = det['class_name']
        
        # 绘制边界框
        draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
        
        # 绘制标签
        label = f"{class_name} {confidence:.2f}"
        draw.text((x1, y1 - 20), label, fill='red')
    
    return image

# 主界面
model = load_model()

uploaded_file = st.file_uploader("上传电缆图片", type=['jpg', 'jpeg', 'png'])

if uploaded_file and model:
    image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("原始图片")
        st.image(image, use_column_width=True)
    
    if st.button("开始检测"):
        with st.spinner("检测中..."):
            detections = simple_detection(image, model)
            result_image = draw_detections(image.copy(), detections)
            
            with col2:
                st.subheader("检测结果")
                st.image(result_image, use_column_width=True)
                
                if detections:
                    st.success(f"发现 {len(detections)} 个缺陷")
                    for det in detections:
                        st.write(f"- {det['class_name']} (置信度: {det['confidence']:.3f})")
                else:
                    st.info("未检测到缺陷")
