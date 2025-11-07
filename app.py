import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import os
from ultralytics import YOLO
import torch

# 页面配置
st.set_page_config(
    page_title="电缆缺陷检测系统",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 标题和描述
st.title("电缆缺陷检测系统")
st.markdown("上传电缆图片，AI自动检测各种缺陷类型")
st.markdown("---")

@st.cache_resource
def load_model():
    """加载PyTorch模型"""
    try:
        # 查找可能的模型文件
        model_files = [f for f in os.listdir('.') if f.endswith('.pt')]
        st.info(f"找到的模型文件: {model_files}")
        
        if not model_files:
            st.error("未找到模型文件 (.pt)")
            st.info("请确保模型文件已上传到正确目录")
            return None
        
        # 选择第一个模型文件
        model_path = model_files[0]
        st.info(f"正在加载模型: {model_path}")
        
        # 加载模型
        model = YOLO(model_path)
        
        st.success("模型加载成功!")
        return model
        
    except Exception as e:
        st.error(f"模型加载失败: {str(e)}")
        return None

# 类别名称映射（根据你的实际模型调整）
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

# 颜色映射
COLORS = [
    (255, 0, 0),      # 红色 - 断裂股线
    (0, 255, 0),      # 绿色 - 焊接股线
    (0, 0, 255),      # 蓝色 - 弯曲股线
    (255, 255, 0),    # 黄色 - 长划痕
    (255, 0, 255),    # 紫色 - 压碎
    (0, 255, 255),    # 青色 - 间隔股线
    (255, 165, 0),    # 橙色 - 沉积物
    (128, 0, 128),    # 深紫色 - 断裂
    (165, 42, 42)     # 棕色 - 雷击损坏
]

def draw_detections_pil(image, results, conf_threshold=0.25):
    """使用PIL绘制检测结果"""
    # 转换为PIL图像
    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image)
    else:
        pil_image = image.copy()
    
    draw = ImageDraw.Draw(pil_image)
    detected_count = 0
    detection_details = []
    
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
                    color = COLORS[class_id % len(COLORS)]
                    
                    # 绘制边界框
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                    
                    # 准备标签文本
                    class_name = CLASS_NAMES.get(class_id, f"类别{class_id}")
                    label = f"{class_name} {confidence:.2f}"
                    
                    # 计算文本尺寸
                    bbox = draw.textbbox((0, 0), label)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                    
                    # 绘制标签背景
                    label_bg = [
                        x1, 
                        max(0, y1 - text_height - 5),
                        x1 + text_width + 10,
                        y1
                    ]
                    draw.rectangle(label_bg, fill=color)
                    
                    # 绘制标签文本
                    text_position = (x1 + 5, max(5, y1 - text_height))
                    draw.text(text_position, label, fill=(255, 255, 255))
                    
                    # 记录检测详情
                    detection_details.append({
                        'class_name': class_name,
                        'confidence': confidence,
                        'bbox': [x1, y1, x2, y2]
                    })
    
    return pil_image, detected_count, detection_details

# 侧边栏配置
with st.sidebar:
    st.header("检测设置")
    
    # 置信度阈值
    confidence_threshold = st.slider(
        "检测置信度阈值", 
        min_value=0.1, 
        max_value=0.9, 
        value=0.25,
        help="调整检测的敏感度，值越小越敏感但可能产生误检"
    )
    
    st.markdown("---")
    st.markdown("### 支持检测的缺陷类型")
    
    for class_id, class_name in CLASS_NAMES.items():
        st.write(f"- {class_name}")
    
    st.markdown("---")
    st.markdown("### 使用说明")
    st.info("1. 上传清晰的电缆图片\n2. 调整置信度阈值\n3. 点击开始检测\n4. 查看检测结果和统计")

# 主界面
st.markdown("## 开始检测")

# 加载模型
model = load_model()

# 文件上传区域
uploaded_file = st.file_uploader(
    "选择电缆图片文件",
    type=['jpg', 'jpeg', 'png'],
    help="支持 JPG、JPEG、PNG 格式，建议图片清晰且包含完整的电缆"
)

if uploaded_file is not None:
    # 显示原图
    try:
        image = Image.open(uploaded_file).convert('RGB')
        image_np = np.array(image)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("原始图片")
            st.image(image, use_column_width=True)
            st.write(f"图片尺寸: {image.size[0]} × {image.size[1]} 像素")
        
        # 检测按钮
        if st.button("开始检测", type="primary", use_container_width=True):
            if model is None:
                st.error("模型未加载，无法进行检测")
            else:
                with st.spinner("AI正在分析图片，检测电缆缺陷..."):
                    try:
                        # 使用YOLO模型进行检测
                        results = model(image_np, conf=confidence_threshold, verbose=False)
                        
                        # 绘制检测结果
                        result_image, detected_count, detection_details = draw_detections_pil(
                            image, results, conf_threshold=confidence_threshold
                        )
                        
                        with col2:
                            st.subheader("检测结果")
                            st.image(result_image, use_column_width=True)
                            
                            # 显示统计信息
                            if detected_count > 0:
                                st.success(f"检测完成！共发现 {detected_count} 个缺陷")
                                
                                # 详细统计
                                with st.expander("查看详细统计", expanded=True):
                                    # 按类别统计
                                    defect_count = {}
                                    for detail in detection_details:
                                        class_name = detail['class_name']
                                        defect_count[class_name] = defect_count.get(class_name, 0) + 1
                                    
                                    st.subheader("缺陷类型统计")
                                    for class_name, count in defect_count.items():
                                        st.write(f"{class_name}: {count} 个")
                                    
                                    st.markdown("---")
                                    st.subheader("详细检测结果")
                                    for i, detail in enumerate(detection_details, 1):
                                        st.write(
                                            f"目标 {i}: {detail['class_name']} "
                                            f"(置信度: {detail['confidence']:.3f})"
                                        )
                                        
                            else:
                                st.info("未检测到明显的电缆缺陷")
                                st.markdown("建议尝试：降低置信度阈值、确保图片清晰且包含电缆、尝试不同的拍摄角度")
                                
                    except Exception as e:
                        st.error(f"检测过程中出现错误: {str(e)}")
                        st.info("请尝试重新上传图片或调整设置")
                        
    except Exception as e:
        st.error(f"图片读取失败: {str(e)}")
        st.info("请检查图片文件是否损坏或尝试其他图片")

elif model is None:
    st.warning("等待模型加载完成...")
else:
    st.info("请上传电缆图片开始检测")

# 页脚信息
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 0.9em; margin-top: 2rem;">
    <p>电缆缺陷检测系统 | 基于YOLOv8深度学习技术</p>
    <p>自动识别电缆表面缺陷，提高检测效率和准确性</p>
</div>
""", unsafe_allow_html=True)
