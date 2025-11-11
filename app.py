import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import os
import subprocess
import sys

# 设置页面
st.set_page_config(page_title="电缆缺陷检测", layout="wide")
st.title("电缆缺陷检测系统")

# 修复libGL问题的函数
def fix_yolo_dependencies():
    """检查和修复YOLO依赖问题"""
    try:
        # 先尝试导入YOLO
        from ultralytics import YOLO
        return True, None
    except ImportError as e:
        error_msg = str(e)
        if "libGL.so.1" in error_msg:
            return False, "libGL"
        else:
            return False, "other"

# 检查依赖
yolo_ready, issue_type = fix_yolo_dependencies()

if not yolo_ready:
    if issue_type == "libGL":
        st.warning("检测到系统依赖缺失，正在尝试修复...")
        
        # 显示修复进度
        with st.spinner("安装系统依赖中..."):
            try:
                # 更新包管理器并安装依赖
                result = subprocess.run(
                    ['apt-get', 'update'], 
                    capture_output=True, 
                    text=True,
                    timeout=60
                )
                
                result2 = subprocess.run(
                    ['apt-get', 'install', '-y', 'libgl1-mesa-glx', 'libglib2.0-0'], 
                    capture_output=True, 
                    text=True,
                    timeout=120
                )
                
                if result2.returncode == 0:
                    st.success("依赖安装成功！请刷新页面重试。")
                    st.balloons()
                else:
                    st.error("依赖安装失败，尝试备用方案...")
                    # 尝试opencv-headless方案
                    try:
                        subprocess.run([
                            sys.executable, '-m', 'pip', 'install', 
                            'opencv-python-headless==4.8.1.78'
                        ], check=True)
                        st.success("备用方案安装成功！请刷新页面。")
                    except:
                        st.error("所有修复方案都失败了，请联系管理员。")
                        
            except Exception as e:
                st.error(f"修复过程中出错: {str(e)}")
        
        st.stop()
    else:
        st.error("YOLO库导入失败，请检查安装。")
        st.stop()

# 如果依赖正常，继续执行后续代码
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 现在安全地导入YOLO
from ultralytics import YOLO
YOLO_AVAILABLE = True

# 设置页面
st.set_page_config(page_title="电缆缺陷检测", layout="wide")
st.title("电缆缺陷检测系统")

# 在导入ultralytics之前设置环境变量
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 现在导入ultralytics
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError as e:
    st.error(f"无法导入YOLO: {e}")
    YOLO_AVAILABLE = False

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
    """使用Ultralytics原生方式加载模型"""
    if not YOLO_AVAILABLE:
        st.error("YOLO库不可用")
        return None
        
    try:
        # 查找模型文件
        model_files = [f for f in os.listdir('.') if f.endswith('.pt')]
        if not model_files:
            st.error("未找到.pt模型文件")
            return None
        
        model_path = model_files[0]
        st.info(f"找到模型文件: {model_path}")
        st.info(f"文件大小: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
        
        # 使用Ultralytics YOLO类直接加载
        model = YOLO(model_path)
        
        # 验证模型加载成功
        if hasattr(model, 'names'):
            st.success(f"模型加载成功! 类别数: {len(model.names)}")
            st.info(f"模型类别: {model.names}")
        else:
            st.success("模型加载成功!")
            
        return model
        
    except Exception as e:
        st.error(f"模型加载失败: {str(e)}")
        # 显示详细错误信息
        import traceback
        with st.expander("查看详细错误信息"):
            st.code(traceback.format_exc())
        return None

def draw_detections(image, results, conf_threshold=0.25):
    """绘制检测结果"""
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

# 侧边栏
with st.sidebar:
    st.header("检测设置")
    confidence_threshold = st.slider(
        "置信度阈值", 0.1, 0.9, 0.25
    )
    
    st.markdown("---")
    st.markdown("支持检测的缺陷类型")
    for class_name in CLASS_NAMES.values():
        st.write(f"- {class_name}")

# 主界面
st.markdown("## 开始检测")

# 加载模型
model = load_model()

uploaded_file = st.file_uploader("上传电缆图片", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("原始图片")
            st.image(image, use_column_width=True)
            st.write(f"图片尺寸: {image.size}")
        
        if st.button("开始检测", type="primary"):
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
                                    st.write("**缺陷统计:**")
                                    for name, count in counts.items():
                                        st.write(f"- {name}: {count}个")
                                    
                                    st.markdown("---")
                                    st.write("**详细结果:**")
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

