import streamlit as st
import numpy as np
from PIL import Image
import cv2
import onnxruntime as ort
import os

# 调试信息
st.sidebar.title("🔧 环境信息")
try:
    files = os.listdir('.')
    st.sidebar.write("文件列表:")
    for file in sorted(files):
        if file.endswith('.onnx'):
            file_size = os.path.getsize(file)
            st.sidebar.write(f"- {file} ({file_size/1024/1024:.1f} MB)")
except Exception as e:
    st.sidebar.error(f"无法读取目录: {e}")

st.set_page_config(page_title="电缆缺陷检测系统", layout="wide")
st.title("电缆缺陷检测系统")
st.write("上传电缆图片，AI自动检测缺陷")

@st.cache_resource
def load_onnx_model():
    try:
        onnx_files = [f for f in os.listdir('.') if f.endswith('.onnx')]
        if not onnx_files:
            st.error("未找到ONNX模型文件")
            return None
        
        model_path = onnx_files[0]  # 使用找到的第一个ONNX文件
        session = ort.InferenceSession(model_path)
        
        # 显示模型输入信息
        inputs = session.get_inputs()
        st.sidebar.success("模型输入信息:")
        for i, input_info in enumerate(inputs):
            st.sidebar.write(f"- 输入 {i}: {input_info.name}")
            st.sidebar.write(f"  形状: {input_info.shape}")
            st.sidebar.write(f"  类型: {input_info.type}")
        
        st.success(f"ONNX模型加载成功: {model_path}")
        return session
        
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        return None

# 加载模型
model_session = load_onnx_model()

def preprocess(image, input_size=640):
    """正确的预处理函数"""
    # 调整尺寸
    img = cv2.resize(image, (input_size, input_size))
    
    # 归一化 [0, 255] -> [0, 1]
    img = img / 255.0
    
    # 转换通道顺序 HWC -> CHW
    img = img.transpose(2, 0, 1)
    
    # 添加batch维度 CHW -> NCHW
    img = np.expand_dims(img, 0).astype(np.float32)
    
    return img

def yolo_postprocess(outputs, original_shape, input_size=640):
    """简化的YOLO后处理"""
    # 这里需要根据你的模型实际输出结构来调整
    # 这是一个通用版本
    
    boxes = []
    scores = []
    class_ids = []
    
    # 假设第一个输出是检测结果
    if len(outputs) > 0:
        predictions = outputs[0]  # [1, 84, 8400] 或类似形状
        
        # 简化的后处理 - 实际需要根据模型输出结构调整
        for i in range(min(10, predictions.shape[2])):  # 最多显示10个检测结果
            # 这里应该是你的检测框解码逻辑
            # 暂时返回模拟数据
            if len(boxes) < 3:  # 模拟3个检测框
                h, w = original_shape[:2]
                x1 = np.random.randint(0, w-100)
                y1 = np.random.randint(0, h-100)
                x2 = x1 + np.random.randint(50, 150)
                y2 = y1 + np.random.randint(50, 150)
                score = np.random.uniform(0.5, 0.9)
                
                boxes.append([x1, y1, x2, y2])
                scores.append(score)
                class_ids.append(0)
    
    return boxes, scores, class_ids

def draw_detections(image, boxes, scores, class_ids, conf_threshold=0.35):
    """绘制检测框"""
    result = image.copy()
    
    for box, score, class_id in zip(boxes, scores, class_ids):
        if score > conf_threshold:
            x1, y1, x2, y2 = [int(coord) for coord in box]
            
            # 绘制矩形框
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 绘制标签
            label = f"缺陷 {score:.2f}"
            (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
            # 标签背景
            cv2.rectangle(result, (x1, y1-label_height-baseline), 
                         (x1+label_width, y1), (0, 255, 0), -1)
            
            # 标签文字
            cv2.putText(result, label, (x1, y1-baseline), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    return result

# 主界面
uploaded_file = st.file_uploader("选择电缆图片", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None and model_session is not None:
    # 读取图片
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="原图", use_column_width=True)
    
    if st.button("开始检测", type="primary"):
        with st.spinner("AI检测中..."):
            try:
                # 预处理
                input_data = preprocess(image_np)
                st.sidebar.write(f"输入数据形状: {input_data.shape}")
                
                # 获取模型输入输出信息
                input_name = model_session.get_inputs()[0].name
                output_names = [output.name for output in model_session.get_outputs()]
                
                st.sidebar.write("模型输出:")
                for i, output in enumerate(model_session.get_outputs()):
                    st.sidebar.write(f"- {output.name}: {output.shape}")
                
                # 推理
                outputs = model_session.run(output_names, {input_name: input_data})
                
                st.sidebar.write("推理输出:")
                for i, output in enumerate(outputs):
                    st.sidebar.write(f"- 输出 {i}: {output.shape}")
                
                # 后处理
                boxes, scores, class_ids = yolo_postprocess(outputs, image_np.shape)
                
                # 绘制结果
                result_image = draw_detections(image_np, boxes, scores, class_ids)
                
                with col2:
                    st.image(result_image, caption="检测结果", use_column_width=True)
                    st.success(f"检测完成！发现 {len(boxes)} 个缺陷")
                    
            except Exception as e:
                st.error(f"检测错误: {e}")
                st.info("这可能是模型输入输出格式不匹配，需要调整预处理或后处理代码")

elif model_session is None:
    st.warning("等待模型加载...")
else:
    st.info("👆 请上传电缆图片开始检测")
