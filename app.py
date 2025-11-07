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

def preprocess(image, target_size=640):
    """预处理图像 - 适配动态输入形状"""
    # 确保图像是3通道
    if len(image.shape) == 3 and image.shape[2] == 4:
        # 如果是4通道RGBA，转换为3通道RGB
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    elif len(image.shape) == 2:
        # 如果是灰度图，转换为3通道
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # 获取原始尺寸
    original_h, original_w = image.shape[:2]
    
    # 计算调整后的尺寸，保持宽高比
    scale = min(target_size / original_w, target_size / original_h)
    new_w = int(original_w * scale)
    new_h = int(original_h * scale)
    
    # 调整尺寸
    resized = cv2.resize(image, (new_w, new_h))
    
    # 创建填充后的图像 (target_size x target_size)
    padded = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    padded[:new_h, :new_w] = resized
    
    # 归一化
    img = padded.astype(np.float32) / 255.0
    
    # 转换通道顺序: HWC to CHW
    img = img.transpose(2, 0, 1)  # 从 [H, W, C] 到 [C, H, W]
    
    # 添加批次维度: [C, H, W] 到 [1, C, H, W]
    img = np.expand_dims(img, 0)
    
    return img, (original_h, original_w), (new_h, new_w), scale

def yolo_postprocess(outputs, original_shape, padded_shape, scale, conf_threshold=0.25):
    """YOLOv8 ONNX输出后处理"""
    try:
        predictions = outputs[0]  # shape: [1, 84, 8400] 或类似
        
        boxes = []
        scores = []
        class_ids = []
        
        # 解析预测结果
        for i in range(predictions.shape[2]):
            prediction = predictions[0, :, i]
            
            # 提取边界框 (x_center, y_center, width, height)
            x_center, y_center, width, height = prediction[0:4]
            
            # 转换为绝对坐标 (在640x640图像上)
            x1 = (x_center - width / 2) 
            y1 = (y_center - height / 2)
            x2 = (x_center + width / 2)
            y2 = (y_center + height / 2)
            
            # 提取类别概率
            class_probs = prediction[4:]  # 从第4个开始是类别概率
            class_id = np.argmax(class_probs)
            confidence = class_probs[class_id]
            
            if confidence > conf_threshold:
                # 调整到填充前的尺寸
                pad_h, pad_w = padded_shape
                x1 = int(x1 * pad_w)
                y1 = int(y1 * pad_h)
                x2 = int(x2 * pad_w)
                y2 = int(y2 * pad_h)
                
                # 调整到原始图像尺寸
                orig_h, orig_w = original_shape
                x1 = int(x1 / scale)
                y1 = int(y1 / scale)
                x2 = int(x2 / scale)
                y2 = int(y2 / scale)
                
                # 确保坐标在图像范围内
                x1 = max(0, min(x1, orig_w))
                y1 = max(0, min(y1, orig_h))
                x2 = max(0, min(x2, orig_w))
                y2 = max(0, min(y2, orig_h))
                
                # 只添加有效的边界框
                if x2 > x1 and y2 > y1 and (x2 - x1) > 5 and (y2 - y1) > 5:
                    boxes.append([x1, y1, x2, y2])
                    scores.append(float(confidence))
                    class_ids.append(int(class_id))
        
        return boxes, scores, class_ids
        
    except Exception as e:
        st.error(f"后处理错误: {e}")
        return [], [], []

# 更简单的后处理函数（备选方案）
def simple_yolo_postprocess(outputs, conf_threshold=0.25):
    """简化的YOLO后处理"""
    try:
        # 假设输出是 [1, 84, 8400] 格式
        predictions = outputs[0]
        
        boxes = []
        scores = []
        class_ids = []
        
        # 直接解析，不进行复杂的坐标转换
        for i in range(min(100, predictions.shape[2])):  # 只处理前100个预测
            prediction = predictions[0, :, i]
            
            # 提取坐标和类别
            if len(prediction) >= 5:
                x_center, y_center, width, height = prediction[0:4]
                class_probs = prediction[4:]
                
                if len(class_probs) > 0:
                    class_id = np.argmax(class_probs)
                    confidence = class_probs[class_id]
                    
                    if confidence > conf_threshold:
                        # 简化的坐标计算（相对坐标）
                        x1 = int((x_center - width / 2) * 640)
                        y1 = int((y_center - height / 2) * 640)
                        x2 = int((x_center + width / 2) * 640)
                        y2 = int((y_center + height / 2) * 640)
                        
                        boxes.append([x1, y1, x2, y2])
                        scores.append(float(confidence))
                        class_ids.append(int(class_id))
        
        return boxes, scores, class_ids
        
    except Exception as e:
        st.error(f"简化后处理错误: {e}")
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
    
    detected_count = 0
    for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
        if score > conf_threshold:
            detected_count += 1
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
            text_y = max(15, y1 - 5)  # 确保文本不会超出图像顶部
            cv2.putText(result, label, (x1, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return result, detected_count

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
    
    use_simple_postprocess = st.checkbox("使用简化后处理", value=True, 
                                       help="如果检测不到目标，尝试切换此选项")
    
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

if uploaded_file is not None and model_session is not None:
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
        st.write(f"图像尺寸: {image_np.shape[1]} x {image_np.shape[0]}")
    
    # 检测按钮
    if st.button("🚀 开始检测", type="primary", use_container_width=True):
        with st.spinner("🔍 AI正在检测电缆缺陷..."):
            try:
                # 获取模型输入信息
                input_name = model_session.get_inputs()[0].name
                
                # 预处理
                input_data, original_shape, padded_shape, scale = preprocess(image_np)
                
                st.info(f"预处理信息:")
                st.info(f"- 原始尺寸: {original_shape}")
                st.info(f"- 填充后尺寸: {padded_shape}") 
                st.info(f"- 缩放比例: {scale:.3f}")
                st.info(f"- 输入数据形状: {input_data.shape}")
                
                # 模型推理
                outputs = model_session.run(None, {input_name: input_data})
                
                # 显示输出信息
                st.info("模型输出:")
                for i, output in enumerate(outputs):
                    st.info(f"输出 {i} 形状: {output.shape}")
                
                # 后处理
                if use_simple_postprocess:
                    boxes, scores, class_ids = simple_yolo_postprocess(
                        outputs, conf_threshold=confidence_threshold
                    )
                    method = "简化后处理"
                else:
                    boxes, scores, class_ids = yolo_postprocess(
                        outputs, original_shape, padded_shape, scale, 
                        conf_threshold=confidence_threshold
                    )
                    method = "标准后处理"
                
                st.info(f"使用 {method}, 检测到 {len(boxes)} 个候选目标")
                
                # 绘制结果
                result_image, detected_count = draw_detections(
                    image_np, boxes, scores, class_ids, 
                    conf_threshold=confidence_threshold
                )
                
                with col2:
                    st.subheader("📊 检测结果")
                    st.image(result_image, use_column_width=True)
                    
                    # 显示统计信息
                    if detected_count > 0:
                        st.success(f"✅ 检测完成！发现 {detected_count} 个缺陷")
                        
                        # 详细统计
                        with st.expander("📈 检测详情"):
                            defect_count = {}
                            for class_id in class_ids:
                                if scores[i] > confidence_threshold:
                                    class_name = CLASS_NAMES.get(class_id, f"类别{class_id}")
                                    defect_count[class_name] = defect_count.get(class_name, 0) + 1
                            
                            for class_name, count in defect_count.items():
                                st.write(f"**{class_name}**: {count} 个")
                    else:
                        st.info("ℹ️ 未检测到明显缺陷")
                        st.info("""
                        建议：
                        1. 降低置信度阈值
                        2. 切换后处理方法
                        3. 尝试不同的图片
                        """)
                        
            except Exception as e:
                st.error(f"❌ 检测过程中出现错误: {str(e)}")
                import traceback
                with st.expander("查看详细错误信息"):
                    st.code(traceback.format_exc())

elif model_session is None:
    st.warning("⚠️ 等待模型加载...")
else:
    st.info("👆 请上传电缆图片开始检测")

# 页脚
st.markdown("---")
st.markdown("""
<style>
.footer {
    text-align: center;
    color: gray;
    font-size: 0.8em;
}
</style>
<div class="footer">
    电缆缺陷检测系统 | 基于YOLOv8和ONNX Runtime
</div>
""", unsafe_allow_html=True)
