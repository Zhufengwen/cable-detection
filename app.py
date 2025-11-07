import streamlit as st
import numpy as np
from PIL import Image
import cv2
import onnxruntime as ort
import os

st.set_page_config(page_title="电缆缺陷检测系统", layout="wide")
st.title("🔌 电缆缺陷检测系统")
st.markdown("上传电缆图片，AI自动检测各种缺陷类型")

@st.cache_resource
def load_onnx_model():
    """加载ONNX模型"""
    try:
        # 查找可能的模型文件
        onnx_files = [f for f in os.listdir('.') if f.endswith('.onnx')]
        st.info(f"找到的ONNX文件: {onnx_files}")
        
        if not onnx_files:
            st.error("❌ 未找到ONNX模型文件")
            return None
        
        model_path = onnx_files[0]
        st.info(f"尝试加载模型: {model_path}")
        
        # 设置ONNX Runtime提供者
        providers = ['CPUExecutionProvider']
        if ort.get_device() == 'GPU':
            providers.insert(0, 'CUDAExecutionProvider')
        
        session = ort.InferenceSession(model_path, providers=providers)
        
        # 显示模型输入信息
        model_inputs = session.get_inputs()
        st.success(f"✅ 模型加载成功!")
        for i, input_info in enumerate(model_inputs):
            st.info(f"输入 {i}: 名称='{input_info.name}', 形状={input_info.shape}, 类型={input_info.type}")
        
        return session
        
    except Exception as e:
        st.error(f"❌ 模型加载失败: {str(e)}")
        return None

def preprocess(image, input_size=640):
    """预处理图像 - 修复维度问题"""
    # 确保图像是3通道
    if len(image.shape) == 3 and image.shape[2] == 4:
        # 如果是4通道RGBA，转换为3通道RGB
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    elif len(image.shape) == 2:
        # 如果是灰度图，转换为3通道
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # 调整尺寸
    img = cv2.resize(image, (input_size, input_size))
    
    # 归一化
    img = img.astype(np.float32) / 255.0
    
    # 转换通道顺序: HWC to CHW
    img = img.transpose(2, 0, 1)  # 从 [H, W, C] 到 [C, H, W]
    
    # 添加批次维度: [C, H, W] 到 [1, C, H, W]
    img = np.expand_dims(img, 0)
    
    return img

def yolo_postprocess(outputs, original_shape, input_size=640, conf_threshold=0.25):
    """YOLOv8 ONNX输出后处理 - 修复版本"""
    try:
        # YOLOv8 ONNX输出格式通常是 [1, 84, 8400]
        predictions = outputs[0]  # shape: [1, 84, 8400]
        
        boxes = []
        scores = []
        class_ids = []
        
        # 计算类别数: 84 = 4(bbox) + 80(classes) 或 13 = 4 + 1 + 8
        num_features = predictions.shape[1]
        if num_features == 84:  # COCO 80类
            num_classes = 80
            has_obj_conf = False
        elif num_features == 13:  # 自定义8类 + obj_conf
            num_classes = 8
            has_obj_conf = True
        else:
            # 自动推断
            num_classes = num_features - 5  # 假设有obj_conf
            has_obj_conf = True
        
        st.info(f"模型输出形状: {predictions.shape}, 类别数: {num_classes}")
        
        # 遍历所有8400个预测
        for i in range(predictions.shape[2]):
            prediction = predictions[0, :, i]
            
            if has_obj_conf:
                # 格式: [x_center, y_center, width, height, obj_conf, class_probs...]
                x_center, y_center, width, height, obj_conf = prediction[0:5]
                class_probs = prediction[5:5+num_classes]
            else:
                # 格式: [x_center, y_center, width, height, class_probs...]
                x_center, y_center, width, height = prediction[0:4]
                class_probs = prediction[4:4+num_classes]
                obj_conf = 1.0  # 如果没有obj_conf，设为1.0
            
            # 转换为像素坐标
            x1 = (x_center - width / 2) * input_size
            y1 = (y_center - height / 2) * input_size
            x2 = (x_center + width / 2) * input_size
            y2 = (y_center + height / 2) * input_size
            
            # 获取类别
            class_id = np.argmax(class_probs)
            class_conf = class_probs[class_id]
            
            # 综合置信度
            confidence = obj_conf * class_conf
            
            if confidence > conf_threshold:
                # 缩放回原图尺寸
                scale_x = original_shape[1] / input_size
                scale_y = original_shape[0] / input_size
                
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)
                
                # 确保坐标在图像范围内
                x1 = max(0, min(x1, original_shape[1]))
                y1 = max(0, min(y1, original_shape[0]))
                x2 = max(0, min(x2, original_shape[1]))
                y2 = max(0, min(y2, original_shape[0]))
                
                # 只添加有效的边界框
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
                    scores.append(float(confidence))
                    class_ids.append(int(class_id))
        
        return boxes, scores, class_ids
        
    except Exception as e:
        st.error(f"后处理错误: {e}")
        return [], [], []

# 类别名称映射
CLASS_NAMES = {
    0: "断裂股线", 1: "焊接股线", 2: "弯曲股线", 3: "长划痕",
    4: "压碎", 5: "间隔股线", 6: "沉积物", 7: "断裂",
    8: "雷击损坏"
}

def draw_detections(image, boxes, scores, class_ids, conf_threshold=0.25):
    """绘制检测结果"""
    result = image.copy()
    
    # 颜色映射
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (255, 165, 0), (128, 0, 128),
        (165, 42, 42)
    ]
    
    for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
        if score > conf_threshold:
            x1, y1, x2, y2 = box
            
            # 选择颜色
            color = colors[class_id % len(colors)]
            
            # 绘制边界框
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 3)
            
            # 准备标签文本
            class_name = CLASS_NAMES.get(class_id, f"类别{class_id}")
            label = f"{class_name} {score:.2f}"
            
            # 计算标签尺寸
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            # 绘制标签背景
            label_bg_y1 = max(0, y1 - label_height - 10)
            label_bg_y2 = y1
            cv2.rectangle(result, 
                         (x1, label_bg_y1),
                         (x1 + label_width, label_bg_y2),
                         color, -1)
            
            # 绘制标签文本
            cv2.putText(result, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return result

# 侧边栏配置
with st.sidebar:
    st.header("⚙️ 检测设置")
    confidence_threshold = st.slider(
        "置信度阈值", 
        min_value=0.1, 
        max_value=0.9, 
        value=0.25,
        help="值越小检测越敏感，但可能产生更多误检"
    )
    
    st.markdown("---")
    st.markdown("### 支持检测的缺陷类型")
    for class_id, class_name in CLASS_NAMES.items():
        st.write(f"• {class_name}")

# 主界面
st.markdown("---")

# 加载模型
model_session = load_onnx_model()

# 文件上传
uploaded_file = st.file_uploader(
    "📤 上传电缆图片", 
    type=['jpg', 'jpeg', 'png'],
    help="支持 JPG、JPEG、PNG 格式"
)

if uploaded_file is not None:
    # 读取并显示原图
    image = Image.open(uploaded_file)
    
    # 转换为numpy数组时确保处理RGBA图像
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    
    image_np = np.array(image)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 原图")
        st.image(image, use_column_width=True)
    
    # 检测按钮
    if st.button("🚀 开始检测", type="primary", use_container_width=True):
        if model_session is None:
            st.error("❌ 模型未加载，无法进行检测")
        else:
            with st.spinner("🔍 AI正在检测电缆缺陷..."):
                try:
                    # 获取模型输入信息
                    input_name = model_session.get_inputs()[0].name
                    input_shape = model_session.get_inputs()[0].shape
                    st.info(f"模型期望输入: {input_name}, 形状: {input_shape}")
                    
                    # 预处理
                    input_data = preprocess(image_np)
                    st.info(f"预处理后形状: {input_data.shape}")
                    
                    # 验证输入形状
                    expected_shape = tuple(input_shape)
                    actual_shape = input_data.shape
                    
                    if actual_shape != expected_shape:
                        st.warning(f"输入形状不匹配: 实际{actual_shape} vs 期望{expected_shape}")
                        # 尝试调整形状
                        if len(actual_shape) == 4 and len(expected_shape) == 4:
                            if actual_shape[1] != expected_shape[1]:
                                st.error(f"通道数不匹配: 实际{actual_shape[1]} vs 期望{expected_shape[1]}")
                                # 如果是通道数问题，尝试转换
                                if actual_shape[1] == 4 and expected_shape[1] == 3:
                                    input_data = input_data[:, :3, :, :]  # 取前3个通道
                                    st.info("已从4通道转换为3通道")
                    
                    # 模型推理
                    outputs = model_session.run(None, {input_name: input_data})
                    
                    # 显示输出信息
                    for i, output in enumerate(outputs):
                        st.info(f"输出 {i} 形状: {output.shape}")
                    
                    # 后处理
                    boxes, scores, class_ids = yolo_postprocess(
                        outputs, 
                        image_np.shape, 
                        conf_threshold=confidence_threshold
                    )
                    
                    # 绘制结果
                    result_image = draw_detections(
                        image_np, boxes, scores, class_ids, 
                        conf_threshold=confidence_threshold
                    )
                    
                    with col2:
                        st.subheader("📊 检测结果")
                        st.image(result_image, use_column_width=True)
                        
                        # 显示统计信息
                        if boxes:
                            st.success(f"✅ 检测完成！发现 {len(boxes)} 个缺陷")
                            
                            # 详细统计
                            with st.expander("📈 检测详情"):
                                defect_count = {}
                                for class_id in class_ids:
                                    class_name = CLASS_NAMES.get(class_id, f"类别{class_id}")
                                    defect_count[class_name] = defect_count.get(class_name, 0) + 1
                                
                                for class_name, count in defect_count.items():
                                    st.write(f"**{class_name}**: {count} 个")
                        else:
                            st.info("ℹ️ 未检测到明显缺陷")
                            
                except Exception as e:
                    st.error(f"❌ 检测过程中出现错误: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

else:
    st.info("👆 请上传电缆图片开始检测")
