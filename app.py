import os
# 第一步：彻底禁用所有图形环境
os.environ['DISPLAY'] = ':0'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 强制使用CPU

# 第二步：只导入基础库
import streamlit as st
import numpy as np
from PIL import Image

# 应用界面
st.title(" 电缆缺陷检测系统")
st.write("上传电缆图片，使用AI自动检测缺陷")

# 第三步：在用户点击检测时才导入YOLO（延迟导入）
def load_model_safely():
    """安全加载模型，避免初始化时的图形调用"""
    try:
        # 在函数内部导入，避免模块级别的图形调用
        from ultralytics import YOLO
        model = YOLO('improvements.pt')
        return model, None
    except Exception as e:
        return None, str(e)

# 文件上传器
uploaded_file = st.file_uploader(
    "选择电缆图片", 
    type=['jpg', 'jpeg', 'png'],
    help="支持 JPG、JPEG、PNG 格式"
)

# 处理上传的文件
if uploaded_file is not None:
    # 显示原图
    image = Image.open(uploaded_file)
    st.image(image, caption="上传的电缆图片", use_column_width=True)
    
    # 检测按钮
    if st.button("开始检测", type="primary"):
        with st.spinner("AI正在检测中..."):
            try:
                # 只有在点击检测时才加载模型
                model, error = load_model_safely()
                
                if model is None:
                    st.error(f" 模型加载失败: {error}")
                    # 提供备选方案
                    st.info(" 建议：尝试使用纯CPU模式或检查模型文件")
                else:
                    st.success(" 模型加载成功！开始检测...")
                    
                    # 转换为numpy数组进行检测
                    image_np = np.array(image)
                    
                    # 使用YOLO进行检测（强制使用CPU）
                    results = model(image_np, conf=0.35, device='cpu')
                    
                    # 显示检测结果
                    if len(results) > 0 and len(results[0].boxes) > 0:
                        num_defects = len(results[0].boxes)
                        st.success(f" 检测完成！共发现 {num_defects} 个缺陷")
                        
                        # 尝试使用PIL绘制检测结果，避免OpenCV图形调用
                        try:
                            # 获取检测结果图片
                            result_img = results[0].plot()  # 这个可能还会调用OpenCV
                            result_img_rgb = Image.fromarray(result_img)
                            st.image(result_img_rgb, caption="检测结果", use_column_width=True)
                        except:
                            # 如果绘图失败，只显示文本结果
                            st.info(" 检测结果（无法显示可视化）：")
                            for i, box in enumerate(results[0].boxes):
                                cls_id = int(box.cls[0])
                                conf = float(box.conf[0])
                                st.write(f"缺陷 {i+1}: 类别 {cls_id}, 置信度 {conf:.2f}")
                    else:
                        st.warning(" 未检测到任何缺陷")
                        
            except Exception as e:
                st.error(f" 检测过程中发生错误: {e}")
                st.info("这可能是因为服务器环境缺少图形库支持")

else:
    st.info(" 请在上方上传电缆图片开始检测")

# 页脚信息
st.markdown("---")
st.caption("电缆缺陷检测系统 | 基于YOLO深度学习模型")

# 调试信息
with st.expander("🔧 环境信息"):
    st.write(f"Python版本: {os.sys.version}")
    st.write(f"当前工作目录: {os.getcwd()}")
    st.write("文件列表:", os.listdir('.'))
