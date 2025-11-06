import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import cv2
import os

# 设置页面标题和布局
st.set_page_config(
    page_title="电缆缺陷检测系统",
    page_icon="",
    layout="wide"
)

# 应用标题
st.title(" 电缆缺陷检测系统")
st.markdown("---")

# 检测置信度阈值
confidence_threshold = 0.35  # 固定值，也可以放在主界面

# 加载模型
@st.cache_resource
def load_model():
    """加载YOLO模型"""
    try:
        model_paths = [
            'improvements.pt',
            'runs/detect/unified_optimized/weights/best.pt',
            'runs/detect/unified_low_memory/weights/best.pt', 
            'runs/detect/unified_cpu/weights/best.pt',
            'runs/detect/unified_minimal_safe/weights/best.pt'
        ]
        
        for model_path in model_paths:
            if os.path.exists(model_path):
                model = YOLO(model_path)
                st.success(f"模型加载成功: {os.path.basename(model_path)}")
                return model
        
        # 如果都没有找到，尝试加载默认名称
        model = YOLO('improvements.pt')
        st.warning("使用默认模型")
        return model
        
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        return None

model = load_model()

# 显示模型信息
if model is not None:
    st.write(f"**检测类别:** {len(model.names)}种")
    # 显示所有类别
    classes_text = " | ".join([f"{class_id}: {class_name}" for class_id, class_name in model.names.items()])
    st.write(f"**类别列表:** {classes_text}")

# 主界面 - 图片上传和检测
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("图片上传")
    
    # 上传图片
    uploaded_file = st.file_uploader(
        "选择电缆图片", 
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="支持 JPG, JPEG, PNG, BMP 格式"
    )
    
    # 显示上传的图片
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="上传的图片", use_container_width=True)
    else:
        st.info("请上传电缆图片")
        image = None

    # 检测按钮放在上传图片下方
    if st.button("开始检测", type="primary", disabled=image is None, use_container_width=True):
        if image is not None and model is not None:
            with st.spinner("正在检测缺陷..."):
                try:
                    # 转换图片格式
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    image_np = np.array(image)
                    
                    # 使用模型进行检测
                    results = model(image_np, conf=confidence_threshold)
                    result = results[0]
                    
                    # 绘制检测结果
                    plotted_image = result.plot()
                    plotted_image_rgb = cv2.cvtColor(plotted_image, cv2.COLOR_BGR2RGB)
                    
                    # 在右侧显示结果图片
                    with col2:
                        st.subheader("检测结果")
                        st.image(plotted_image_rgb, caption="检测结果", use_container_width=True)
                        
                        # 统计检测结果
                        detection_stats = {}
                        detections_info = []
                        
                        if result.boxes is not None and len(result.boxes) > 0:
                            for box in result.boxes:
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                conf = box.conf[0].cpu().numpy()
                                class_id = int(box.cls[0].cpu().numpy())
                                
                                class_name = model.names[class_id]
                                
                                if class_name not in detection_stats:
                                    detection_stats[class_name] = 0
                                detection_stats[class_name] += 1
                                
                                detections_info.append({
                                    'class_name': class_name,
                                    'confidence': conf,
                                    'bbox': [x1, y1, x2, y2]
                                })
                        
                        # 显示检测统计
                        st.markdown("### 检测统计")
                        
                        if detection_stats:
                            total_detections = sum(detection_stats.values())
                            st.success(f"发现 {total_detections} 个缺陷")
                            
                            # 显示缺陷类型统计
                            st.markdown("**缺陷类型统计:**")
                            for defect_type, count in detection_stats.items():
                                st.write(f"- {defect_type}: {count}个")
                            
                            # 显示详细检测信息
                            st.markdown("**详细检测信息:**")
                            for i, det in enumerate(detections_info, 1):
                                bbox = det['bbox']
                                st.write(f"{i}. **{det['class_name']}**")
                                st.write(f"   置信度: {det['confidence']:.3f}")
                                st.write(f"   位置: [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]")
                                
                        else:
                            st.success("未发现缺陷")
                            
                        # 添加下载结果的功能
                        if detection_stats:
                            st.download_button(
                                label="下载检测结果图片",
                                data=cv2.imencode('.jpg', cv2.cvtColor(plotted_image_rgb, cv2.COLOR_RGB2BGR))[1].tobytes(),
                                file_name="detection_result.jpg",
                                mime="image/jpeg",
                                use_container_width=True
                            )
                            
                except Exception as e:
                    st.error(f"检测过程中出现错误: {e}")
                    
        elif model is None:
            st.error("模型未加载，无法进行检测")
        else:
            st.error("请先上传图片")

# 如果还没有开始检测，在右侧显示说明
if 'plotted_image_rgb' not in locals():
    with col2:
        st.subheader("使用说明")
        st.markdown("""
        1. **上传图片**: 在左侧上传电缆图片
        2. **开始检测**: 点击"开始检测"按钮
        3. **查看结果**: 检测结果将显示在本区域
        
        ### 支持的缺陷类型:
        - 断股 (broken strand)
        - 焊股 (welded strand) 
        - 弯股 (bent strand)
        - 长划痕 (long scratch)
        - 压扁 (crushed)
        - 间距股 (spaced strand)
        - 沉积物 (deposit)
        - 断裂 (break)
        - 雷击 (thunderbolt)
        """)


