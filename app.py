import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import os
from ultralytics import YOLO
import torch

# 在导入任何其他库之前设置环境变量
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '0'
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
os.environ['CV_IO_MAX_IMAGE_PIXELS'] = '0'

# 禁用OpenCV的多线程
os.environ['OPENCV_FOR_THREADS_NUM'] = '1'

st.set_page_config(
    page_title="电缆缺陷检测系统",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("电缆缺陷检测系统")
st.markdown("上传电缆图片，AI自动检测各种缺陷类型")
st.markdown("---")

@st.cache_resource
def load_model():
    """加载PyTorch模型"""
    try:
        model_files = [f for f in os.listdir('.') if f.endswith('.pt')]
        if not model_files:
            st.error("未找到模型文件 (.pt)")
            return None
        
        model_path = model_files[0]
        model = YOLO(model_path)
        st.success("模型加载成功!")
        return model
        
    except Exception as e:
        st.error(f"模型加载失败: {str(e)}")
        return None

# 类别名称映射
CLASS_NAMES = {
    0: "断裂股线", 1: "焊接股线", 2: "弯曲股线", 3: "长划痕",
    4: "压碎", 5: "间隔股线", 6: "沉积物", 7: "断裂", 8: "雷击损坏"
}

COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (255, 165, 0), (128, 0, 128),
    (165, 42, 42)
]

def draw_detections_pil(image, results, conf_threshold=0.25):
    """使用PIL绘制检测结果"""
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
                    
                    color = COLORS[class_id % len(COLORS)]
                    class_name = CLASS_NAMES.get(class_id, f"类别{class_id}")
                    label = f"{class_name} {confidence:.2f}"
                    
                    # 绘制边界框
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                    
                    # 绘制标签
                    bbox = draw.textbbox((0, 0), label)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                    
                    label_bg = [
                        x1, 
                        max(0, y1 - text_height - 5),
                        x1 + text_width + 10,
                        y1
                    ]
                    draw.rectangle(label_bg, fill=color)
                    draw.text((x1 + 5, max(5, y1 - text_height)), label, fill=(255, 255, 255))
                    
                    detection_details.append({
                        'class_name': class_name,
                        'confidence': confidence,
                        'bbox': [x1, y1, x2, y2]
                    })
    
    return pil_image, detected_count, detection_details

# 侧边栏
with st.sidebar:
    st.header("检测设置")
    confidence_threshold = st.slider(
        "检测置信度阈值", 0.1, 0.9, 0.25,
        help="值越小检测越敏感但可能产生误检"
    )
    
    st.markdown("---")
    st.markdown("支持检测的缺陷类型")
    for class_name in CLASS_NAMES.values():
        st.write(f"- {class_name}")

# 主界面
model = load_model()

uploaded_file = st.file_uploader(
    "选择电缆图片文件",
    type=['jpg', 'jpeg', 'png']
)

if uploaded_file is not None and model is not None:
    try:
        image = Image.open(uploaded_file).convert('RGB')
        image_np = np.array(image)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("原始图片")
            st.image(image, use_column_width=True)
        
        if st.button("开始检测", type="primary"):
            with st.spinner("AI正在分析图片..."):
                try:
                    # 使用YOLO进行检测
                    results = model(image_np, conf=confidence_threshold, verbose=False)
                    
                    # 绘制结果
                    result_image, detected_count, detection_details = draw_detections_pil(
                        image, results, confidence_threshold
                    )
                    
                    with col2:
                        st.subheader("检测结果")
                        st.image(result_image, use_column_width=True)
                        
                        if detected_count > 0:
                            st.success(f"检测完成！发现 {detected_count} 个缺陷")
                            with st.expander("详细统计"):
                                defect_count = {}
                                for detail in detection_details:
                                    class_name = detail['class_name']
                                    defect_count[class_name] = defect_count.get(class_name, 0) + 1
                                
                                for class_name, count in defect_count.items():
                                    st.write(f"{class_name}: {count} 个")
                        else:
                            st.info("未检测到明显的电缆缺陷")
                            
                except Exception as e:
                    st.error(f"检测失败: {str(e)}")
                    
    except Exception as e:
        st.error(f"图片读取失败: {str(e)}")

elif model is None:
    st.warning("等待模型加载...")
else:
    st.info("请上传电缆图片开始检测")
