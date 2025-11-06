import streamlit as st
import numpy as np
from PIL import Image
import cv2
import onnxruntime as ort

st.set_page_config(page_title="电缆缺陷检测系统", layout="wide")
st.title("电缆缺陷检测系统")
st.write("上传电缆图片，AI自动检测缺陷")

# 加载ONNX模型 - 这个不会调用libGL！
@st.cache_resource
def load_onnx_model():
    try:
        # 提供多个可能的模型路径
        model_paths = ['improvements.onnx', 'improvement.onnx', 'best.onnx']
        for model_path in model_paths:
            try:
                session = ort.InferenceSession(model_path)
                st.success(f"ONNX模型加载成功: {model_path}")
                return session
            except:
                continue
        st.error("未找到可用的ONNX模型文件")
        return None
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        return None

# 加载模型
model_session = load_onnx_model()

# YOLO的预处理和后处理函数
def preprocess(image, input_size=640):
    """预处理图像"""
    img = cv2.resize(image, (input_size, input_size))
    img = img / 255.0
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0).astype(np.float32)
    return img

def postprocess(outputs, image_shape, conf_threshold=0.35):
    """后处理检测结果"""
    # 这里需要根据你的模型输出结构进行调整
    # 这是一个简化的示例
    boxes = []
    scores = []
    class_ids = []
    
    # 实际实现需要根据你的ONNX模型输出结构来写
    # 这里只是示例逻辑
    return boxes, scores, class_ids

uploaded_file = st.file_uploader("选择电缆图片", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None and model_session is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="原图", use_column_width=True)
    
    if st.button("开始检测", type="primary"):
        with st.spinner("AI检测中..."):
            try:
                # 转换为numpy数组
                image_np = np.array(image)
                
                # 预处理
                input_data = preprocess(image_np)
                
                # 推理
                outputs = model_session.run(None, {model_session.get_inputs()[0].name: input_data})
                
                # 后处理
                boxes, scores, class_ids = postprocess(outputs, image_np.shape)
                
                # 绘制结果
                result_image = image_np.copy()
                for box, score, class_id in zip(boxes, scores, class_ids):
                    if score > 0.35:  # 置信度阈值
                        x1, y1, x2, y2 = box
                        cv2.rectangle(result_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(result_image, f'Defect {score:.2f}', (int(x1), int(y1)-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                with col2:
                    st.image(result_image, caption="检测结果", use_column_width=True)
                    st.success(f"检测完成！发现 {len(boxes)} 个缺陷")
                    
            except Exception as e:
                st.error(f"检测错误: {e}")

else:
    if model_session is None:
        st.warning("等待模型加载...")
    else:
        st.info("请上传电缆图片开始检测")
