import streamlit as st
import numpy as np
from PIL import Image
import os

python
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
# 这些设置帮助OpenCV在无头环境中运行
os.environ['DISPLAY'] = ':0'

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

# 基础设置
st.set_page_config(page_title="电缆缺陷检测系统", layout="wide")
st.title("电缆缺陷检测系统")

# 检查依赖
try:
    from ultralytics import YOLO
    import cv2
    deps_loaded = True
except ImportError as e:
    st.error(f"依赖加载失败: {e}")
    deps_loaded = False

if not deps_loaded:
    st.stop()

# 加载模型
@st.cache_resource
def load_model():
    try:
        if os.path.exists('improvements.pt'):
            return YOLO('improvements.pt')
        return None
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        return None

model = load_model()

# 界面代码
uploaded_file = st.file_uploader("上传电缆图片", type=['jpg', 'jpeg', 'png'])

if uploaded_file and model:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="原图", use_container_width=True)
    
    if st.button("开始检测"):
        with st.spinner("检测中..."):
            try:
                results = model(np.array(image), conf=0.35)
                result_img = results[0].plot()
                result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                
                with col2:
                    st.image(result_img_rgb, caption="检测结果", use_container_width=True)
                    st.success("检测完成！")
            except Exception as e:
                st.error(f"检测错误: {e}")
