import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import os
import sys

# 设置环境变量 - 必须在最前面
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

# 设置页面
st.set_page_config(page_title="电缆缺陷检测", layout="wide")
st.title("电缆缺陷检测系统")

# 显示环境状态
st.sidebar.markdown("## 系统状态")

# 尝试绕过OpenCV导入YOLO
YOLO_AVAILABLE = False
model = None

try:
    # 设置环境变量来禁用OpenCV的GUI功能
    os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
    os.environ['CV_IO_ENABLE_OPENEXR'] = '1'
    
    # 尝试直接导入YOLO
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    st.sidebar.success("YOLO库可用")
    
except ImportError as e:
    error_msg = str(e)
    st.sidebar.error("YOLO库不可用")
    
    # 显示详细错误信息
    with st.expander("查看错误详情"):
        st.code(error_msg)
    
    # 提供解决方案
    st.error("依赖问题解决方案")
    
    st.markdown("请确保您的项目包含以下文件：")
    
    st.markdown("**1. requirements.txt**")
    st.code("""
ultralytics==8.0.186
Pillow>=10.0.0
numpy>=1.21.0
streamlit>=1.28.0
torch>=2.0.0 --index-url https://download.pytorch.org/whl/cpu
torchvision>=0.15.0 --index-url https://download.pytorch.org/whl/cpu
""")
    
    st.markdown("**2. packages.txt**")
    st.code("""
libgl1
libglib2.0-0
""")
    
    st.markdown("**3. 重新部署应用**")
    
    # 停止执行后续代码
    st.stop()

# 类别映射
CLASS_NAMES = {
    0: "断裂股线", 1: "焊接股线", 2: "弯曲股线", 3: "长划痕",
    4: "压碎", 5: "间隔股线", 6: "沉积物", 7: "断裂", 8: "雷击损坏"
}

COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (255, 165, 0), (128, 0, 128),
    (165, 42, 42)
]

@st.cache_resource
def load_model():
    """加载YOLO模型"""
    try:
        # 查找模型文件
        model_files = [f for f in os.listdir('.') if f.endswith('.pt')]
        model_files.extend([f for f in os.listdir('/mount/src/cable-detection') if f.endswith('.pt')])
        
        if not model_files:
            st.error("未找到.pt模型文件")
            st.info("请确保模型文件已上传到应用根目录")
            return None
        
        model_path = model_files[0]
        st.sidebar.info(f"找到模型: {model_path}")
        
        # 显示文件信息
        try:
            file_size = os.path.getsize(model_path) / (1024*1024)
            st.sidebar.info(f"文件大小: {file_size:.2f} MB")
        except:
            pass
        
        # 加载模型
        model = YOLO(model_path)
        
        # 验证模型加载成功
        if hasattr(model, 'names'):
            st.sidebar.success("模型加载成功!")
            st.sidebar.info(f"类别数: {len(model.names)}")
        else:
            st.sidebar.success("模型加载成功!")
            
        return model
        
    except Exception as e:
        st.sidebar.error("模型加载失败")
        with st.expander("查看详细错误"):
            st.code(str(e))
        return None

def draw_detections(image, results, conf_threshold=0.25):
    """绘制检测结果 - 使用纯PIL"""
    drawable_image = image.copy()
    draw = ImageDraw.Draw(drawable_image)
    detections = []
    
    for result in results:
        if hasattr(result, 'boxes'):
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    confidence = box.conf.item()
                    if confidence > conf_threshold:
                        class_id = int(box.cls.item())
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        
                        class_name = CLASS_NAMES.get(class_id, f"类别{class_id}")
                        color = COLORS[class_id % len(COLORS)]
                        
                        # 绘制边界框
                        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                        
                        # 绘制标签
                        label = f"{class_name} {confidence:.2f}"
                        bbox = draw.textbbox((0, 0), label)
                        text_width = bbox[2] - bbox[0]
                        text_height = bbox[3] - bbox[1]
                        
                        # 标签背景
                        label_bg = [
                            x1, 
                            max(0, y1 - text_height - 10),
                            x1 + text_width + 10, 
                            y1
                        ]
                        draw.rectangle(label_bg, fill=color)
                        draw.text((x1 + 5, max(5, y1 - text_height - 5)), label, fill=(255, 255, 255))
                        
                        detections.append({
                            'class_name': class_name,
                            'confidence': confidence,
                            'bbox': [int(x1), int(y1), int(x2), int(y2)]
                        })
    
    return drawable_image, detections

# 侧边栏设置
with st.sidebar:
    st.header("检测设置")
    confidence_threshold = st.slider(
        "置信度阈值", 0.1, 0.9, 0.25, 0.05,
        help="值越高，检测要求越严格"
    )
    
    st.markdown("---")
    st.markdown("支持检测的缺陷类型")
    for class_name in CLASS_NAMES.values():
        st.write(f"- {class_name}")

# 主界面
st.markdown("## 开始检测")

# 加载模型
if YOLO_AVAILABLE:
    model = load_model()
else:
    model = None
    st.error("YOLO不可用，无法加载模型")

# 文件上传
uploaded_file = st.file_uploader(
    "上传电缆图片", 
    type=['jpg', 'jpeg', 'png'],
    help="支持 JPG, JPEG, PNG 格式"
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("原始图片")
            st.image(image, use_column_width=True)
            st.write(f"图片尺寸: {image.size}")
        
        if st.button("开始检测", type="primary", disabled=model is None):
            if model is None:
                st.error("模型未加载，无法检测")
            else:
                with st.spinner("AI检测中..."):
                    try:
                        # 使用模型进行推理
                        results = model(image, conf=confidence_threshold, verbose=False)
                        
                        # 绘制结果
                        result_image, detections = draw_detections(image, results, confidence_threshold)
                        
                        with col2:
                            st.subheader("检测结果")
                            st.image(result_image, use_column_width=True)
                            
                            if detections:
                                st.success(f"检测完成！发现 {len(detections)} 个缺陷")
                                
                                with st.expander("检测详情"):
                                    # 统计
                                    from collections import Counter
                                    counts = Counter([d['class_name'] for d in detections])
                                    st.write("缺陷统计:")
                                    for name, count in counts.items():
                                        st.write(f"- {name}: {count}个")
                                    
                                    st.markdown("---")
                                    st.write("详细结果:")
                                    for i, det in enumerate(detections, 1):
                                        st.write(f"{i}. {det['class_name']} - 置信度: {det['confidence']:.3f}")
                            else:
                                st.info("未检测到缺陷")
                                
                    except Exception as e:
                        st.error(f"检测失败: {str(e)}")
                        
    except Exception as e:
        st.error(f"图片处理失败: {str(e)}")

else:
    st.info("请上传电缆图片开始检测")

# 部署成功提示
if model is not None and YOLO_AVAILABLE:
    st.success("系统运行正常！")
