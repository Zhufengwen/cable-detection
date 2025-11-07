import torch
torch.serialization.add_safe_globals(["ultralytics.nn.tasks.DetectionModel"])
import streamlit as st
from PIL import Image, ImageDraw
import torch
import numpy as np
import os

st.set_page_config(page_title="电缆缺陷检测", layout="wide")
st.title("电缆缺陷检测系统")

# 类别颜色映射
COLORS = {
    '断裂股线': (255, 0, 0),      # 红色
    '焊接股线': (0, 255, 0),      # 绿色  
    '弯曲股线': (0, 0, 255),      # 蓝色
    '长划痕': (255, 255, 0),      # 黄色
    '压碎': (255, 0, 255),        # 紫色
    '间隔股线': (0, 255, 255),    # 青色
    '沉积物': (255, 165, 0),      # 橙色
    '断裂': (128, 0, 128),        # 深紫色
    '雷击损坏': (165, 42, 42)     # 棕色
}

# 类别ID到名称的映射
CLASS_NAMES = {
    0: "断裂股线", 
    1: "焊接股线", 
    2: "弯曲股线", 
    3: "长划痕",
    4: "压碎", 
    5: "间隔股线", 
    6: "沉积物", 
    7: "断裂",
    8: "雷击损坏"
}

@st.cache_resource
def load_model():
    """加载训练好的模型"""
    try:
        model_files = [f for f in os.listdir('.') if f.endswith('.pt')]
        if not model_files:
            st.error("未找到模型文件")
            return None
        
        model_path = model_files[0]
        st.info(f"正在加载模型: {model_path}")
        
        # 直接使用torch加载模型
        model = torch.load(model_path, map_location='cpu', weights_only=False)
        model.eval()  # 设置为评估模式
        
        st.success("模型加载成功!")
        return model
        
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        return None

def draw_detections_on_image(image, detections):
    """在图像上绘制检测框和标签"""
    drawable_image = image.copy()
    draw = ImageDraw.Draw(drawable_image)
    
    for detection in detections:
        class_name = detection['class_name']
        confidence = detection['confidence']
        bbox = detection['bbox']  # [x1, y1, x2, y2]
        
        # 获取颜色
        color = COLORS.get(class_name, (255, 0, 0))
        
        # 绘制边界框
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # 准备标签文本
        label = f"{class_name} {confidence:.2f}"
        
        # 计算文本尺寸
        bbox_text = draw.textbbox((0, 0), label)
        text_width = bbox_text[2] - bbox_text[0]
        text_height = bbox_text[3] - bbox_text[1]
        
        # 绘制标签背景
        label_bg = [
            x1, 
            max(0, y1 - text_height - 10),
            x1 + text_width + 10, 
            y1
        ]
        draw.rectangle(label_bg, fill=color)
        
        # 绘制标签文本
        text_position = (x1 + 5, max(5, y1 - text_height - 5))
        draw.text(text_position, label, fill=(255, 255, 255))
    
    return drawable_image

def detect_with_model(image, model, confidence_threshold=0.25):
    """使用模型进行真实检测"""
    try:
        # 将PIL图像转换为numpy数组
        image_np = np.array(image)
        
        # 这里需要根据你的模型结构进行推理
        # 由于我不知道你的具体模型结构，这里提供一个通用模板
        
        # 方法1: 如果模型是标准的YOLO格式
        if hasattr(model, 'predict'):
            # 使用模型的predict方法
            results = model.predict(image_np, conf=confidence_threshold)
            detections = []
            
            for result in results:
                if hasattr(result, 'boxes'):
                    boxes = result.boxes
                    for box in boxes:
                        conf = box.conf.item()
                        cls = int(box.cls.item())
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        
                        detections.append({
                            'class_name': CLASS_NAMES.get(cls, f"类别{cls}"),
                            'confidence': conf,
                            'bbox': [int(x1), int(y1), int(x2), int(y2)]
                        })
            return detections
        
        # 方法2: 如果是自定义模型结构
        else:
            # 这里需要根据你的模型具体结构来编写推理代码
            # 你需要告诉我你的模型是如何输出的，我才能写正确的推理逻辑
            st.warning("使用通用推理方法，可能需要调整")
            
            # 模拟真实推理（这里需要你提供模型的具体输出格式）
            # 临时返回空列表，避免错误
            return []
            
    except Exception as e:
        st.error(f"推理错误: {e}")
        return []

# 侧边栏配置
with st.sidebar:
    st.header("检测设置")
    confidence_threshold = st.slider(
        "置信度阈值", 
        min_value=0.1, 
        max_value=0.9, 
        value=0.25
    )
    
    st.markdown("---")
    st.markdown("支持检测的缺陷类型")
    for class_name in CLASS_NAMES.values():
        st.write(f"- {class_name}")

# 主界面
model = load_model()

uploaded_file = st.file_uploader("上传电缆图片", type=['jpg', 'jpeg', 'png'])

if uploaded_file and model is not None:
    image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("原始图片")
        st.image(image, use_column_width=True)
        st.write(f"图像尺寸: {image.size[0]} × {image.size[1]}")
    
    if st.button("开始检测", type="primary"):
        with st.spinner("AI正在分析图像..."):
            # 使用真实模型进行检测
            detections = detect_with_model(image, model, confidence_threshold)
            
            if detections:
                # 绘制检测结果
                result_image = draw_detections_on_image(image, detections)
                
                with col2:
                    st.subheader("检测结果")
                    st.image(result_image, use_column_width=True)
                    
                    st.success(f"检测完成！发现 {len(detections)} 个缺陷")
                    
                    with st.expander("查看检测详情"):
                        # 统计信息
                        from collections import Counter
                        class_counts = Counter([det['class_name'] for det in detections])
                        
                        st.write("**缺陷统计:**")
                        for class_name, count in class_counts.items():
                            st.write(f"- {class_name}: {count} 个")
                        
                        st.markdown("---")
                        st.write("**详细结果:**")
                        for i, detection in enumerate(detections, 1):
                            st.write(f"{i}. {detection['class_name']} - 置信度: {detection['confidence']:.3f}")
            else:
                with col2:
                    st.subheader("检测结果")
                    st.image(image, use_column_width=True)
                    st.info("未检测到任何缺陷")

elif model is None:
    st.warning("模型加载失败，无法进行检测")
else:
    st.info("请上传电缆图片开始检测")


