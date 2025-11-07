import streamlit as st
import numpy as np
from PIL import Image
import cv2
import torch
from ultralytics import YOLO
import os

st.set_page_config(page_title="电缆缺陷检测系统", layout="wide")
st.title("电缆缺陷检测系统")
st.markdown("上传电缆图片，AI自动检测各种缺陷类型")

@st.cache_resource
def load_model():
    """加载PyTorch模型"""
    try:
        # 查找可能的模型文件
        model_files = [f for f in os.listdir('.') if f.endswith('.pt')]
        st.info(f"找到的模型文件: {model_files}")
        
        if not model_files:
            st.error("未找到模型文件 (.pt)")
            return None
        
        # 选择第一个模型文件
        model_path = model_files[0]
        st.info(f"尝试加载模型: {model_path}")
        
        # 加载模型
        model = YOLO(model_path)
        
        st.success(f"模型加载成功!")
        st.info(f"模型类别数: {model.model.model[-1].nc}")
        
        return model
        
    except Exception as e:
        st.error(f"模型加载失败: {str(e)}")
        return None

# 类别名称映射（根据你的模型调整）
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

def draw_detections(image, results, conf_threshold=0.25):
    """绘制检测结果"""
    result = image.copy()
    detected_count = 0
    
    # 颜色映射
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (255, 165, 0), (128, 0, 128),
        (165, 42, 42)
    ]
    
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                confidence = box.conf[0].cpu().numpy()
                if confidence > conf_threshold:
                    detected_count += 1
                    class_id = int(box.cls[0].cpu().numpy())
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    
                    # 选择颜色
                    color = colors[class_id % len(colors)]
                    
                    # 绘制边界框
                    cv2.rectangle(result, (x1, y1), (x2, y2), color, 3)
                    
                    # 准备标签文本
                    class_name = CLASS_NAMES.get(class_id, f"类别{class_id}")
                    label = f"{class_name} {confidence:.2f}"
                    
                    # 计算标签尺寸
                    (label_width, label_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    
                    # 绘制标签背景
                    label_bg_y1 = max(0, y1 - label_height - 10)
                    label_bg_y2 = y1
                    cv2.rectangle(result, 
                                 (x1, label_bg_y1),
                                 (x1 + label_width, label_bg_y2),
                                 color, -1)
                    
                    # 绘制标签文本
                    text_y = max(15, y1 - 5)
                    cv2.putText(result, label, (x1, text_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return result, detected_count

# 侧边栏配置
with st.sidebar:
    st.header("⚙️ 检测设置")
    confidence_threshold = st.slider(
        "置信度阈值", 
        min_value=0.1, 
        max_value=0.9, 
        value=0.25,
        help="值越小检测越敏感，但可能产生更多误检"
    )
    
    st.markdown("---")
    st.markdown("### 支持检测的缺陷类型")
    for class_id, class_name in CLASS_NAMES.items():
        st.write(f"• {class_name}")

# 主界面
st.markdown("---")

# 加载模型
model = load_model()

# 文件上传
uploaded_file = st.file_uploader(
    "上传电缆图片", 
    type=['jpg', 'jpeg', 'png'],
    help="支持 JPG、JPEG、PNG 格式"
)

if uploaded_file is not None and model is not None:
    # 读取并显示原图
    image = Image.open(uploaded_file)
    
    # 转换为numpy数组时确保处理RGBA图像
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    
    image_np = np.array(image)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("原图")
        st.image(image, use_column_width=True)
        st.write(f"图像尺寸: {image_np.shape[1]} x {image_np.shape[0]}")
    
    # 检测按钮
    if st.button("开始检测", type="primary", use_container_width=True):
        with st.spinner("AI正在检测电缆缺陷..."):
            try:
                # 使用YOLO模型进行检测
                results = model(image_np, conf=confidence_threshold, verbose=False)
                
                # 绘制结果
                result_image, detected_count = draw_detections(
                    image_np, results, conf_threshold=confidence_threshold
                )
                
                with col2:
                    st.subheader("检测结果")
                    st.image(result_image, use_column_width=True)
                    
                    # 显示统计信息
                    if detected_count > 0:
                        st.success(f"检测完成！发现 {detected_count} 个缺陷")
                        
                        # 详细统计
                        with st.expander("📈 检测详情"):
                            defect_count = {}
                            for result in results:
                                boxes = result.boxes
                                if boxes is not None:
                                    for box in boxes:
                                        if box.conf[0] > confidence_threshold:
                                            class_id = int(box.cls[0].cpu().numpy())
                                            class_name = CLASS_NAMES.get(class_id, f"类别{class_id}")
                                            defect_count[class_name] = defect_count.get(class_name, 0) + 1
                            
                            for class_name, count in defect_count.items():
                                st.write(f"**{class_name}**: {count} 个")
                                
                            # 显示每个检测的详细信息
                            st.markdown("---")
                            st.markdown("**详细检测结果:**")
                            for i, result in enumerate(results):
                                boxes = result.boxes
                                if boxes is not None:
                                    for j, box in enumerate(boxes):
                                        if box.conf[0] > confidence_threshold:
                                            class_id = int(box.cls[0].cpu().numpy())
                                            class_name = CLASS_NAMES.get(class_id, f"类别{class_id}")
                                            confidence = box.conf[0].cpu().numpy()
                                            st.write(f"目标 {j+1}: {class_name} (置信度: {confidence:.3f})")
                    else:
                        st.info("未检测到明显缺陷")
                        st.info("""
                        建议：
                        1. 降低置信度阈值
                        2. 确保图片清晰且包含电缆
                        3. 尝试不同的图片角度
                        """)
                        
            except Exception as e:
                st.error(f"检测过程中出现错误: {str(e)}")
                import traceback
                with st.expander("查看详细错误信息"):
                    st.code(traceback.format_exc())

elif model is None:
    st.warning("等待模型加载...")
else:
    st.info("请上传电缆图片开始检测")

# 页脚
st.markdown("---")
st.markdown("""
<style>
.footer {
    text-align: center;
    color: gray;
    font-size: 0.8em;
}
</style>
<div class="footer">
    电缆缺陷检测系统 | 基于YOLOv8
</div>
""", unsafe_allow_html=True)
