import os
# 必须在所有导入之前设置环境变量
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ['DISPLAY'] = ':0'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import streamlit as st
import numpy as np
from PIL import Image
import cv2
from ultralytics import YOLO
import tempfile

# 页面基础设置
st.set_page_config(
    page_title="电缆缺陷检测系统",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 应用标题和描述
st.title(" 电缆缺陷检测系统")
st.markdown("---")
st.markdown("上传电缆图片，使用YOLO模型自动检测缺陷")

# 侧边栏
with st.sidebar:
    st.header("设置")
    confidence = st.slider("检测置信度", 0.1, 1.0, 0.35, 0.05)
    st.markdown("---")
    st.info("""
    **使用说明：**
    1. 上传电缆图片
    2. 调整检测置信度
    3. 点击开始检测
    4. 查看检测结果
    """)

# 检查依赖加载状态
@st.cache_resource
def check_dependencies():
    """检查所有依赖是否正常加载"""
    dependencies_status = {}
    try:
        import cv2
        dependencies_status['opencv'] = True
    except Exception as e:
        dependencies_status['opencv'] = False
        dependencies_status['opencv_error'] = str(e)
    
    try:
        from ultralytics import YOLO
        dependencies_status['ultralytics'] = True
    except Exception as e:
        dependencies_status['ultralytics'] = False
        dependencies_status['ultralytics_error'] = str(e)
    
    try:
        import numpy as np
        dependencies_status['numpy'] = True
    except Exception as e:
        dependencies_status['numpy'] = False
        dependencies_status['numpy_error'] = str(e)
    
    return dependencies_status

# 加载模型
@st.cache_resource
def load_model():
    """加载YOLO模型"""
    try:
        model_paths = ['improvements.pt', 'improvement.pt']
        for model_path in model_paths:
            if os.path.exists(model_path):
                model = YOLO(model_path)
                st.sidebar.success(f"模型加载成功: {model_path}")
                return model
        
        st.sidebar.error(" 未找到模型文件")
        return None
        
    except Exception as e:
        st.sidebar.error(f" 模型加载失败: {str(e)}")
        return None

# 显示依赖状态
deps_status = check_dependencies()

if not all([deps_status['opencv'], deps_status['ultralytics'], deps_status['numpy']]):
    st.error(" 依赖加载失败")
    with st.expander("查看详细错误"):
        if not deps_status['opencv']:
            st.error(f"OpenCV错误: {deps_status.get('opencv_error', 'Unknown error')}")
        if not deps_status['ultralytics']:
            st.error(f"Ultralytics错误: {deps_status.get('ultralytics_error', 'Unknown error')}")
        if not deps_status['numpy']:
            st.error(f"NumPy错误: {deps_status.get('numpy_error', 'Unknown error')}")
    st.stop()

# 加载模型
model = load_model()
if model is None:
    st.error("无法加载检测模型，请确保模型文件存在")
    st.stop()

# 主界面
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader(" 上传图片")
    uploaded_file = st.file_uploader(
        "选择电缆图片", 
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="支持 JPG, JPEG, PNG, BMP 格式"
    )

# 处理上传的图片
if uploaded_file is not None:
    try:
        # 读取图片
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        
        with col1:
            st.image(image, caption=f"原图 | 尺寸: {image.size}", use_container_width=True)
            
            # 检测按钮
            if st.button(" 开始检测", type="primary", use_container_width=True):
                with st.spinner("AI正在检测电缆缺陷..."):
                    try:
                        # 使用YOLO进行检测
                        results = model(image_np, conf=confidence, verbose=False)
                        
                        with col2:
                            st.subheader(" 检测结果")
                            
                            if len(results) > 0 and len(results[0].boxes) > 0:
                                # 绘制检测结果
                                result_img = results[0].plot()
                                result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                                
                                st.image(result_img_rgb, caption="检测结果", use_container_width=True)
                                
                                # 显示检测统计
                                boxes = results[0].boxes
                                num_detections = len(boxes)
                                
                                st.success(f" 检测完成！共发现 {num_detections} 个缺陷")
                                
                                # 显示详细信息
                                with st.expander(" 详细检测结果"):
                                    for i, box in enumerate(boxes):
                                        cls = int(box.cls[0])
                                        conf = float(box.conf[0])
                                        st.write(f"缺陷 {i+1}: 类别 {cls}, 置信度 {conf:.2f}")
                                
                            else:
                                st.warning(" 未检测到任何缺陷")
                                # 显示原图
                                st.image(image, caption="未发现缺陷", use_container_width=True)
                                
                    except Exception as e:
                        st.error(f" 检测过程中发生错误: {str(e)}")
                        
    except Exception as e:
        st.error(f" 图片处理错误: {str(e)}")

else:
    with col2:
        st.info(" 请在左侧上传电缆图片开始检测")

# 页脚
st.markdown("---")
st.caption("电缆缺陷检测系统 | 基于 YOLO 深度学习模型")

# 调试信息（可选）
with st.sidebar:
    with st.expander(" 调试信息"):
        st.write(f"OpenCV版本: {cv2.__version__}")
        st.write(f"NumPy版本: {np.__version__}")
        st.write("依赖状态:", deps_status)
