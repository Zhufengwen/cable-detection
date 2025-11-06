import os
# 第一步：设置环境变量（在导入任何库之前！）
os.environ['DISPLAY'] = ':0'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# 第二步：导入库
import streamlit as st
import numpy as np
from PIL import Image

# 第三步：安全地导入OpenCV
try:
    import cv2
    # 设置OpenCV为优化模式
    cv2.setUseOptimized(True)
    OPENCV_AVAILABLE = True
except Exception as e:
    OPENCV_AVAILABLE = False
    st.warning(f"OpenCV图形功能不可用，但基础功能正常")

# 第四步：导入YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception as e:
    YOLO_AVAILABLE = False
    st.error(f"YOLO加载失败: {e}")

# 应用界面
st.title(" 电缆缺陷检测系统")
st.write("上传电缆图片，使用AI自动检测缺陷")

# 第五步：加载模型
@st.cache_resource
def load_model():
    try:
        model = YOLO('improvements.pt')
        st.success(" 模型加载成功！")
        return model
    except Exception as e:
        st.error(f" 模型加载失败: {e}")
        return None

model = load_model()

# 第六步：文件上传器
uploaded_file = st.file_uploader(
    "选择电缆图片", 
    type=['jpg', 'jpeg', 'png'],
    help="支持 JPG、JPEG、PNG 格式"
)

# 第七步：处理上传的文件
if uploaded_file is not None:
    # 安全地处理图片
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    
    # 显示原图 - 使用 use_column_width 而不是 use_container_width
    st.image(image, caption="上传的电缆图片", use_column_width=True)
    
    # 检测按钮
    if st.button("开始检测", type="primary"):
        with st.spinner("AI正在检测中..."):
            try:
                if model is not None and YOLO_AVAILABLE:
                    # 使用YOLO进行检测
                    results = model(image_np, conf=0.35)
                    
                    # 显示检测结果
                    if len(results) > 0 and len(results[0].boxes) > 0:
                        st.success(f" 检测完成！共发现 {len(results[0].boxes)} 个缺陷")
                        
                        # 如果有OpenCV，显示带检测框的图片
                        if OPENCV_AVAILABLE:
                            result_img = results[0].plot()
                            result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                            st.image(result_img_rgb, caption="检测结果", use_column_width=True)
                        else:
                            st.info("检测完成，但无法显示带框图片")
                    else:
                        st.warning(" 未检测到任何缺陷")
                        
            except Exception as e:
                st.error(f" 检测过程中发生错误: {e}")

elif model is None:
    st.warning(" 请等待模型加载完成...")
else:
    st.info(" 请在上方上传电缆图片开始检测")

# 页脚信息
st.markdown("---")
st.caption("电缆缺陷检测系统 | 基于YOLO深度学习模型")
