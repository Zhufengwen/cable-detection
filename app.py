import streamlit as st
from PIL import Image, ImageDraw
import os

st.set_page_config(page_title="电缆缺陷检测", layout="wide")
st.title("电缆缺陷检测系统")

# 类别颜色映射
COLORS = {
    '断裂股线': (255, 0, 0),      # 红色
    '焊接股线': (0, 255, 0),      # 绿色  
    '弯曲股线': (0, 0, 255),      # 蓝色
    '长划痕': (255, 255, 0),      # 黄色
    '压碎': (255, 0, 255),        # 紫色
    '间隔股线': (0, 255, 255),    # 青色
    '沉积物': (255, 165, 0),      # 橙色
    '断裂': (128, 0, 128),        # 深紫色
    '雷击损坏': (165, 42, 42)     # 棕色
}

def draw_detections_on_image(image, detections):
    """在图像上绘制检测框和标签"""
    # 创建可绘制的图像副本
    drawable_image = image.copy()
    draw = ImageDraw.Draw(drawable_image)
    
    for detection in detections:
        class_name = detection['class_name']
        confidence = detection['confidence']
        bbox = detection['bbox']  # [x1, y1, x2, y2]
        
        # 获取颜色
        color = COLORS.get(class_name, (255, 0, 0))  # 默认红色
        
        # 绘制边界框
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # 准备标签文本
        label = f"{class_name} {confidence:.2f}"
        
        # 计算文本尺寸
        bbox_text = draw.textbbox((0, 0), label)
        text_width = bbox_text[2] - bbox_text[0]
        text_height = bbox_text[3] - bbox_text[1]
        
        # 绘制标签背景
        label_bg = [
            x1, 
            max(0, y1 - text_height - 10),  # 确保不会超出图像顶部
            x1 + text_width + 10, 
            y1
        ]
        draw.rectangle(label_bg, fill=color)
        
        # 绘制标签文本
        text_position = (x1 + 5, max(5, y1 - text_height - 5))
        draw.text(text_position, label, fill=(255, 255, 255))
    
    return drawable_image

# 模拟检测函数 - 现在包含真实的边界框坐标
def mock_detection(image):
    """模拟检测结果"""
    width, height = image.size
    
    # 根据图像尺寸生成合理的边界框位置
    return [
        {
            'class_name': '断裂股线', 
            'confidence': 0.85, 
            'bbox': [
                int(width * 0.1),   # x1
                int(height * 0.2),  # y1  
                int(width * 0.4),   # x2
                int(height * 0.5)   # y2
            ]
        },
        {
            'class_name': '压碎', 
            'confidence': 0.72, 
            'bbox': [
                int(width * 0.6),   # x1
                int(height * 0.3),  # y1
                int(width * 0.9),   # x2
                int(height * 0.7)   # y2
            ]
        },
        {
            'class_name': '长划痕',
            'confidence': 0.68,
            'bbox': [
                int(width * 0.3),
                int(height * 0.1),
                int(width * 0.7),
                int(height * 0.25)
            ]
        }
    ]

# 文件上传
uploaded_file = st.file_uploader("上传电缆图片", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    # 读取图像
    image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("原始图片")
        st.image(image, use_column_width=True)
        st.write(f"图像尺寸: {image.size[0]} × {image.size[1]}")
    
    # 检测按钮
    if st.button("开始检测", type="primary"):
        with st.spinner("AI检测中..."):
            # 执行模拟检测
            detections = mock_detection(image)
            
            # 在图像上绘制检测结果
            result_image = draw_detections_on_image(image, detections)
            
            with col2:
                st.subheader("检测结果")
                st.image(result_image, use_column_width=True)
                
                # 显示检测统计
                if detections:
                    st.success(f"检测完成！发现 {len(detections)} 个缺陷")
                    
                    # 显示详细信息
                    with st.expander("查看检测详情"):
                        # 按类别统计
                        from collections import Counter
                        class_counts = Counter([det['class_name'] for det in detections])
                        
                        st.write("**缺陷统计:**")
                        for class_name, count in class_counts.items():
                            st.write(f"- {class_name}: {count} 个")
                        
                        st.markdown("---")
                        st.write("**详细检测结果:**")
                        for i, detection in enumerate(detections, 1):
                            st.write(
                                f"{i}. {detection['class_name']} - "
                                f"置信度: {detection['confidence']:.3f} - "
                                f"位置: {detection['bbox']}"
                            )
                else:
                    st.info("未检测到任何缺陷")

else:
    st.info("请上传电缆图片开始检测")

# 页脚说明
st.markdown("---")
st.markdown("电缆缺陷检测演示系统 | 基于模拟检测算法")
