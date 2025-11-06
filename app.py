import streamlit as st
import numpy as np
from PIL import Image
import cv2
import onnxruntime as ort
import os

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
        
        model_path = onnx_files[0]
        session = ort.InferenceSession(model_path)
        st.success(f"ONNX模型加载成功: {model_path}")
        return session
        
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        return None

# 加载模型
model_session = load_onnx_model()

def preprocess(image, input_size=640):
    """预处理图像"""
    img = cv2.resize(image, (input_size, input_size))
    img = img / 255.0
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0).astype(np.float32)
    return img

def yolo_postprocess(outputs, original_shape, input_size=640, conf_threshold=0.35):
    """YOLOv8后处理 - 针对 (1, 13, 8400) 输出格式"""
    predictions = outputs[0]  # [1, 13, 8400]
    
    boxes = []
    scores = []
    class_ids = []
    
    # 输出形状: (1, 13, 8400)
    # 13 = 4(bbox) + 1(置信度) + 8(类别数) - 根据你的类别数调整
    num_classes = 8  # 根据你的模型调整这个数字
    
    # 遍历所有8400个预测
    for i in range(predictions.shape[2]):
        prediction = predictions[0, :, i]  # 获取单个预测 [13]
        
        # 提取置信度 (通常是第4个元素，但需要确认)
        objectness = prediction[4]
        
        if objectness > conf_threshold:
            # 提取边界框坐标 (cx, cy, w, h 格式)
            cx, cy, w, h = prediction[0:4]
            
            # 转换为 (x1, y1, x2, y2) 格式
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            
            # 提取类别概率
            class_probs = prediction[5:5+num_classes]
            class_id = np.argmax(class_probs)
            class_score = class_probs[class_id]
            
            # 综合得分 = 目标置信度 * 类别置信度
            score = objectness * class_score
            
            if score > conf_threshold:
                # 缩放回原图尺寸
                scale_x = original_shape[1] / input_size
                scale_y = original_shape[0] / input_size
                
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)
                
                boxes.append([x1, y1, x2, y2])
                scores.append(score)
                class_ids.append(class_id)
    
    return boxes, scores, class_ids

def draw_detections(image, boxes, scores, class_ids, conf_threshold=0.35):
    """绘制检测框"""
    result = image.copy()
    
    for box, score, class_id in zip(boxes, scores, class_ids):
        if score > conf_threshold:
            x1, y1, x2, y2 = box
            
            # 绘制矩形框
            color = (0, 255, 0)  # 绿色
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签
            label = f"缺陷 {score:.2f}"
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            
            # 标签背景
            cv2.rectangle(result, 
                         (x1, y1 - label_height - baseline),
                         (x1 + label_width, y1),
                         color, -1)
            
            # 标签文字
            cv2.putText(result, label, (x1, y1 - baseline),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
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
                
                # 获取模型输入名称
                input_name = model_session.get_inputs()[0].name
                
                # 推理
                outputs = model_session.run(None, {input_name: input_data})
                
                # 后处理
                boxes, scores, class_ids = yolo_postprocess(outputs, image_np.shape)
                
                # 绘制结果
                result_image = draw_detections(image_np, boxes, scores, class_ids)
                
                with col2:
                    st.image(result_image, caption="检测结果", use_column_width=True)
                    
                    if len(boxes) > 0:
                        st.success(f"检测完成！发现 {len(boxes)} 个缺陷")
                        
                        # 显示检测详情
                        with st.expander("检测详情"):
                            for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
                                st.write(f"缺陷 {i+1}: 置信度 {score:.3f}, 位置 {box}")
                    else:
                        st.info("未检测到缺陷")
                        
            except Exception as e:
                st.error(f"检测错误: {e}")

elif model_session is None:
    st.warning("等待模型加载...")
else:
    st.info("请上传电缆图片开始检测")
