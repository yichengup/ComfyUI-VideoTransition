"""
视频立方体转场节点 - Playwright版本（批处理优化）
两个视频片段之间的3D立方体旋转转场效果，支持批处理和内存优化
"""

import torch
import json
import cv2
import numpy as np
import math
import time
from comfy.comfy_types.node_typing import ComfyNodeABC, InputTypeDict, IO


class VideoCubeTransitionNode(ComfyNodeABC):
    """视频立方体转场 - 两个视频之间的3D立方体旋转转场（批处理优化版）"""
    
    CATEGORY = "VideoTransition"
    
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "video1": (IO.IMAGE,),  # 第一个视频（立方体正面）
                "video2": (IO.IMAGE,),  # 第二个视频（立方体另一面）
                "direction": ([
                    "horizontal",  # 水平立方体转场（左右转）
                    "vertical",    # 垂直立方体转场（上下转）
                ],),
                "total_frames": (IO.INT, {"default": 60, "min": 4, "max": 300}),
                "fps": (IO.INT, {"default": 30, "min": 15, "max": 60}),
            },
            "optional": {
                "rotation_angle": (IO.FLOAT, {"default": 90.0, "min": 45.0, "max": 180.0}),
                "cube_size": (IO.FLOAT, {"default": 300.0, "min": 200.0, "max": 600.0}),
                "perspective": (IO.FLOAT, {"default": 800.0, "min": 500.0, "max": 2000.0}),
                "use_gpu": (IO.BOOLEAN, {"default": False}),
                "batch_size": (IO.INT, {"default": 5, "min": 1, "max": 20}),
                "background_color": (IO.STRING, {"default": "#000000"}),
                "width": (IO.INT, {"default": 640, "min": 320, "max": 1920}),
                "height": (IO.INT, {"default": 640, "min": 240, "max": 1080}),
            }
        }
    
    RETURN_TYPES = (IO.IMAGE,)
    RETURN_NAMES = ("video",)
    FUNCTION = "generate_cube_transition"
    
    async def generate_cube_transition(
        self, 
        video1, 
        video2,
        direction,
        total_frames, 
        fps,
        rotation_angle=90.0,
        cube_size=300.0,
        perspective=800.0,
        use_gpu=False,
        batch_size=5,
        background_color="#000000",
        width=640,
        height=640
    ):
        """生成视频立方体转场效果 - 批处理优化版本"""
        
        start_time = time.time()
        print(f"Starting cube transition: {direction}, {total_frames} frames")
        
        # 提取视频帧
        frames1 = self._extract_video_frames(video1)
        frames2 = self._extract_video_frames(video2)
        print(f"Video frames: {len(frames1)} -> {len(frames2)}, generating {total_frames} transition frames")
        
        # 🎯 立方体面尺寸直接使用画布尺寸（矩形面）
        cube_width = width
        cube_height = height
        
        # 解析背景颜色
        bg_color_bgr = self._parse_background_color(background_color)
        
        # 预计算优化：提前计算所有帧的base64数据
        precompute_start = time.time()
        
        frame1_base64_list = []
        frame2_base64_list = []
        
        for i in range(total_frames):
            progress = i / (total_frames - 1) if total_frames > 1 else 0
            
            # 获取对应的帧
            if len(frames1) == 1:
                frame1_data = self._tensor_to_numpy(frames1[0])
            else:
                frame1_idx = min(int(progress * (len(frames1) - 1)), len(frames1) - 1)
                frame1_data = self._tensor_to_numpy(frames1[frame1_idx])
            
            if len(frames2) == 1:
                frame2_data = self._tensor_to_numpy(frames2[0])
            else:
                frame2_idx = min(int(progress * (len(frames2) - 1)), len(frames2) - 1)
                frame2_data = self._tensor_to_numpy(frames2[frame2_idx])
            
            # 预计算base64数据
            frame1_base64_list.append(self._numpy_to_base64(frame1_data))
            frame2_base64_list.append(self._numpy_to_base64(frame2_data))
        
        precompute_time = time.time() - precompute_start
        
        # 使用Playwright直接渲染3D效果
        from playwright.async_api import async_playwright
        
        playwright = await async_playwright().start()
        
        try:
            # 根据GPU设置选择渲染方式
            if use_gpu:
                print("Playwright browser starting with GPU acceleration")
                # GPU硬件加速
                browser = await playwright.chromium.launch(
                    headless=True,
                    args=[
                        '--enable-gpu',                    # 启用GPU
                        '--use-gl=angle',                  # 使用ANGLE（支持GPU）
                        '--enable-webgl',
                        '--enable-accelerated-2d-canvas',
                        '--disable-dev-shm-usage',
                        '--hide-scrollbars',
                        '--mute-audio',
                        '--no-sandbox',                    # 避免权限问题
                        '--disable-setuid-sandbox',
                    ]
                )
            else:
                print("Playwright browser starting with SwiftShader (CPU rendering)")
                # SwiftShader软件渲染
                browser = await playwright.chromium.launch(
                    headless=True,
                    args=[
                        '--use-angle=swiftshader',         # 强制使用CPU软件渲染
                        '--enable-webgl',
                        '--enable-accelerated-2d-canvas',
                        '--disable-dev-shm-usage',
                        '--hide-scrollbars',
                        '--mute-audio',
                    ]
                )
            
            # 生成HTML模板（传递cube_width、cube_height和direction）
            html_content = self._generate_html_template(
                direction, rotation_angle, cube_width, cube_height, perspective, background_color, width, height
            )
            
            # 创建页面
            page = await browser.new_page(viewport={'width': width, 'height': height})
            
            try:
                # 加载HTML页面
                await page.set_content(html_content, wait_until='domcontentloaded', timeout=30000)
                
                # 等待页面初始化
                await page.wait_for_function("window.cubeController && window.cubeController.ready", timeout=5000)
                
                # 使用生成器进行内存优化
                output_frames = []
                render_start = time.time()
                
                for batch_start in range(0, total_frames, batch_size):
                    batch_end = min(batch_start + batch_size, total_frames)
                    batch_indices = list(range(batch_start, batch_end))
                    
                    # 批处理：一次处理多个帧
                    batch_frames = await self._process_batch(
                        page, batch_indices, frame1_base64_list, frame2_base64_list, total_frames, rotation_angle
                    )
                    
                    # 立即添加到结果中，避免内存积累
                    output_frames.extend(batch_frames)
                    
                    # 显示进度 - 每2个批次或完成时显示
                    if batch_end % (batch_size * 2) == 0 or batch_end == total_frames:
                        progress = (batch_end / total_frames) * 100
                        elapsed = time.time() - render_start
                        if batch_end < total_frames:
                            eta = (elapsed / batch_end) * (total_frames - batch_end)
                            print(f"Processing frames {batch_end}/{total_frames} ({progress:.1f}%) - ETA: {eta:.1f}s")
                        else:
                            print(f"Rendering completed: {batch_end}/{total_frames} frames in {elapsed:.2f}s")
                
                render_time = time.time() - render_start
            
            finally:
                await page.close()
            
            await browser.close()
        finally:
            await playwright.stop()
        
        # 转换为tensor
        video_tensor = self._frames_to_tensor(output_frames)
        
        total_time = time.time() - start_time
        print(f"Cube transition completed: {video_tensor.shape} in {total_time:.2f}s")
        
        return (video_tensor,)
    
    async def _process_batch(self, page, batch_indices, frame1_base64_list, frame2_base64_list, total_frames, rotation_angle):
        """批处理渲染多个帧"""
        batch_frames = []
        
        for i in batch_indices:
            progress = i / (total_frames - 1) if total_frames > 1 else 0
            
            # 使用预计算的base64数据
            frame1_base64 = frame1_base64_list[i]
            frame2_base64 = frame2_base64_list[i]
            
            # 更新立方体面的背景图片和旋转
            await page.evaluate(f"""
                const frontFace = document.getElementById('frontFace');
                const secondFace = document.getElementById('secondFace');
                
                if (frontFace) {{
                    frontFace.style.backgroundImage = 'url({frame1_base64})';
                }}
                
                if (secondFace) {{
                    secondFace.style.backgroundImage = 'url({frame2_base64})';
                }}
                
                // 更新旋转
                if (window.cubeController) {{
                    window.cubeController.updateRotation({progress});
                }}
            """)
            
            # 优化等待时间
            await page.wait_for_timeout(20)  # 从200ms减少到20ms
            
            # 强制触发重绘
            await page.evaluate("document.body.offsetHeight")
            
            # 等待CSS动画完成
            await page.wait_for_timeout(10)  # 从50ms减少到10ms
            
            # 优化截图：使用JPEG格式
            screenshot_bytes = await page.screenshot(type='jpeg', quality=85)
            
            # 转换为numpy数组
            from PIL import Image
            import io
            image = Image.open(io.BytesIO(screenshot_bytes)).convert('RGB')
            final_frame = np.array(image).astype(np.uint8)
            
            batch_frames.append(final_frame)
        
        return batch_frames
    
    def _generate_html_template(self, direction, rotation_angle, cube_width, cube_height, perspective, background_color, width, height):
        """生成HTML模板 - 矩形立方体面（支持水平和垂直方向）"""
        
        # 根据方向设置立方体面的变换
        if direction == "horizontal":
            # 水平立方体（左右转）：正面 + 右侧面
            # 深度使用宽度的一半
            cube_depth = cube_width / 2
            front_transform = f"translateZ({cube_depth}px)"
            second_face_transform = f"rotateY(90deg) translateZ({cube_depth}px)"
            second_face_class = "right"
        else:
            # 垂直立方体（上下转）：正面 + 底面
            # 深度使用高度的一半（重要！）
            cube_depth = cube_height / 2
            front_transform = f"translateZ({cube_depth}px)"
            second_face_transform = f"rotateX(90deg) translateZ({cube_depth}px)"
            second_face_class = "bottom"
        
        return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            width: {width}px;
            height: {height}px;
            background: {background_color};
            overflow: hidden;
            display: flex;
            justify-content: center;
            align-items: center;
            perspective: {perspective}px;
            perspective-origin: 50% 50%;
        }}
        
        .cube-container {{
            width: {cube_width}px;
            height: {cube_height}px;
            position: relative;
            transform-style: preserve-3d;
            transform: rotateY(0deg);
        }}
        
        .cube-face {{
            position: absolute;
            width: {cube_width + 1}px;
            height: {cube_height}px;
            backface-visibility: hidden;
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            border: none;
            left: -0.5px;
        }}
        
        .front {{
            transform: {front_transform};
        }}
        
        .{second_face_class} {{
            transform: {second_face_transform};
        }}
    </style>
</head>
<body>
    <div class="cube-container" id="cubeContainer">
        <div class="cube-face front" id="frontFace"></div>
        <div class="cube-face {second_face_class}" id="secondFace"></div>
    </div>
    
    <script>
        class CubeController {{
            constructor() {{
                this.container = document.getElementById('cubeContainer');
                this.frontFace = document.getElementById('frontFace');
                this.secondFace = document.getElementById('secondFace');
                this.cubeWidth = {cube_width};
                this.cubeHeight = {cube_height};
                this.cubeDepth = {cube_depth};
                this.rotationAngle = {rotation_angle};
                this.direction = '{direction}';
                this.currentAngle = 0;
                this.ready = true;
                
                // console.log('🎬 CubeController initialized, direction=' + this.direction + ', size=' + this.cubeWidth + 'x' + this.cubeHeight);
            }}
            
            updateRotation(progress) {{
                // progress: 0 -> 1
                const angle = progress * this.rotationAngle;
                this.currentAngle = angle;
                
                // 🎯 优化缩放曲线：开始和结束填满画布，中间缩小显示立方体3D效果
                // 使用平滑的sin曲线：1.0 -> 0.65 -> 1.0
                let scale;
                if (progress < 0.05) {{
                    // 阶段1（0 -> 0.05）：保持完整显示
                    scale = 1.0;
                }} else if (progress > 0.95) {{
                    // 阶段4（0.95 -> 1.0）：恢复完整显示
                    scale = 1.0;
                }} else {{
                    // 阶段2-3（0.05 -> 0.95）：使用sin曲线平滑缩放
                    // 将progress映射到0-π范围，sin(x)从0到1再到0
                    const t = (progress - 0.05) / 0.9;  // 归一化到0-1
                    const sinValue = Math.sin(t * Math.PI);  // 0 -> 1 -> 0
                    scale = 1.0 - (sinValue * 0.35);  // 1.0 -> 0.65 -> 1.0
                }}
                
                // 根据方向选择旋转轴
                let rotateTransform;
                if (this.direction === 'horizontal') {{
                    // 水平立方体：绕Y轴旋转（左右转）
                    rotateTransform = `rotateY(${{-angle}}deg)`;
                }} else {{
                    // 垂直立方体：绕X轴旋转（上下转）
                    rotateTransform = `rotateX(${{-angle}}deg)`;
                }}
                
                // 同时应用旋转和缩放
                const transform = `${{rotateTransform}} scale(${{scale}})`;
                this.container.style.transform = transform;
                
                // 强制重绘
                void this.container.offsetHeight;
                
                // console.log(`[JS] Update: direction=${{this.direction}}, angle=${{angle.toFixed(1)}}°, scale=${{scale.toFixed(2)}}, transform=${{transform}}`);
            }}
            
            getFaceTransforms() {{
                const frontRect = this.frontFace.getBoundingClientRect();
                const secondRect = this.secondFace.getBoundingClientRect();
                
                // 获取计算后的变换矩阵
                const frontStyle = window.getComputedStyle(this.frontFace);
                const secondStyle = window.getComputedStyle(this.secondFace);
                
                return {{
                    front: {{
                        rect: {{
                            x: frontRect.x,
                            y: frontRect.y,
                            width: frontRect.width,
                            height: frontRect.height
                        }},
                        transform: frontStyle.transform,
                        visible: this._isFaceVisible(this.frontFace),
                        zIndex: this._calculateZIndex(frontStyle.transform)
                    }},
                    second: {{
                        rect: {{
                            x: secondRect.x,
                            y: secondRect.y,
                            width: secondRect.width,
                            height: secondRect.height
                        }},
                        transform: secondStyle.transform,
                        visible: this._isFaceVisible(this.secondFace),
                        zIndex: this._calculateZIndex(secondStyle.transform)
                    }},
                    currentAngle: this.currentAngle
                }};
            }}
            
            _isFaceVisible(element) {{
                const style = window.getComputedStyle(element);
                const transform = style.transform;
                
                if (transform === 'none') return true;
                
                const matrix = this._parseMatrix(transform);
                if (!matrix) return true;
                
                // 3D矩阵的背面剔除
                if (matrix.length === 16) {{
                    // matrix3d: 检查Z轴方向
                    const m11 = matrix[0], m12 = matrix[1], m13 = matrix[2];
                    const m21 = matrix[4], m22 = matrix[5], m23 = matrix[6];
                    const m31 = matrix[8], m32 = matrix[9], m33 = matrix[10];
                    
                    // 计算法向量的Z分量
                    const normalZ = m31;
                    return normalZ >= -0.1; // 面向观察者
                }}
                
                return true;
            }}
            
            _calculateZIndex(transformStr) {{
                if (transformStr === 'none') return 0;
                
                const matrix = this._parseMatrix(transformStr);
                if (!matrix) return 0;
                
                // 返回Z轴的translateZ值
                if (matrix.length === 16) {{
                    return matrix[14]; // matrix3d的Z平移
                }}
                
                return 0;
            }}
            
            _parseMatrix(matrixString) {{
                if (matrixString === 'none') return null;
                const values = matrixString.match(/matrix.*?\\((.+)\\)/);
                if (!values) return null;
                return values[1].split(/,\\s*/).map(parseFloat);
            }}
        }}
        
        window.cubeController = new CubeController();
        // console.log('✅ CubeController initialized');
    </script>
</body>
</html>
"""
    
    def _compose_frame_with_opencv(self, frame1_data, frame2_data, cube_transforms, bg_color_bgr, width, height, cube_size):
        """使用OpenCV合成最终帧 - 使用透视变换"""
        
        # 创建背景画布
        canvas = np.full((height, width, 3), bg_color_bgr, dtype=np.uint8)
        
        # 调整视频帧大小到立方体尺寸
        frame1_resized = cv2.resize(frame1_data, (cube_size, cube_size), interpolation=cv2.INTER_AREA)
        frame2_resized = cv2.resize(frame2_data, (cube_size, cube_size), interpolation=cv2.INTER_AREA)
        
        # 绘制立方体的两个面
        faces_to_draw = []
        
        # 正面（video1）
        front_transform = cube_transforms.get('front', {})
        if front_transform.get('visible', True):
            faces_to_draw.append({
                'frame': frame1_resized,
                'transform': front_transform,
                'name': 'front'
            })
        
        # 右侧面（video2）
        right_transform = cube_transforms.get('right', {})
        if right_transform.get('visible', True):
            faces_to_draw.append({
                'frame': frame2_resized,
                'transform': right_transform,
                'name': 'right'
            })
        
        # 按Z值排序（先绘制远的面）
        faces_to_draw.sort(key=lambda x: x['transform'].get('zIndex', 0))
        
        # 绘制所有面
        for face_data in faces_to_draw:
            frame = face_data['frame']
            transform = face_data['transform']
            rect = transform.get('rect', {})
            
            x = rect.get('x', 0)
            y = rect.get('y', 0)
            w = rect.get('width', cube_size)
            h = rect.get('height', cube_size)
            
            # 简单的边界检查
            if w <= 0 or h <= 0:
                continue
            
            # 直接使用rect的尺寸进行缩放（Playwright已经计算好透视）
            try:
                # 缩放到目标尺寸
                frame_scaled = cv2.resize(frame, (int(w), int(h)), interpolation=cv2.INTER_LINEAR)
                
                # 计算绘制位置
                draw_x = int(x)
                draw_y = int(y)
                draw_w = int(w)
                draw_h = int(h)
                
                # 裁剪到画布范围内
                src_x1 = 0
                src_y1 = 0
                src_x2 = draw_w
                src_y2 = draw_h
                
                dst_x1 = draw_x
                dst_y1 = draw_y
                dst_x2 = draw_x + draw_w
                dst_y2 = draw_y + draw_h
                
                # 处理负坐标
                if dst_x1 < 0:
                    src_x1 = -dst_x1
                    dst_x1 = 0
                if dst_y1 < 0:
                    src_y1 = -dst_y1
                    dst_y1 = 0
                
                # 处理超出边界
                if dst_x2 > width:
                    src_x2 -= (dst_x2 - width)
                    dst_x2 = width
                if dst_y2 > height:
                    src_y2 -= (dst_y2 - height)
                    dst_y2 = height
                
                # 确保有有效的绘制区域
                if dst_x2 > dst_x1 and dst_y2 > dst_y1 and src_x2 > src_x1 and src_y2 > src_y1:
                    canvas[dst_y1:dst_y2, dst_x1:dst_x2] = frame_scaled[src_y1:src_y2, src_x1:src_x2]
            except Exception as e:
                print(f"Warning: Error drawing face {face_data['name']}: {e}")
                pass
        
        return canvas
    
    def _calculate_z_index(self, face_transform):
        """计算面的Z索引（用于深度排序）"""
        transform_str = face_transform.get('transform', 'none')
        
        if transform_str == 'none':
            return 0
        
        # 简单提取translateZ值
        if 'translateZ' in transform_str:
            try:
                z_value = float(transform_str.split('translateZ(')[1].split('px)')[0])
                return z_value
            except:
                return 0
        
        return 0
    
    def _extract_video_frames(self, video_tensor):
        """从视频tensor中提取帧"""
        frames = []
        
        if video_tensor.dim() == 4:
            batch_size = video_tensor.shape[0]
            for i in range(batch_size):
                frame = video_tensor[i]
                frames.append(frame)
        else:
            frames.append(video_tensor)
        
        return frames
    
    def _tensor_to_numpy(self, frame_tensor):
        """将tensor转换为numpy数组"""
        if isinstance(frame_tensor, torch.Tensor):
            frame_np = frame_tensor.cpu().numpy()
        else:
            frame_np = frame_tensor
        
        # 转换数据类型
        if frame_np.dtype == np.float32 or frame_np.dtype == np.float64:
            frame_np = (frame_np * 255).clip(0, 255).astype(np.uint8)
        
        return frame_np
    
    def _numpy_to_base64(self, frame_np):
        """将numpy数组转换为base64字符串"""
        import io
        import base64
        from PIL import Image
        
        # 确保是uint8类型
        if frame_np.dtype != np.uint8:
            frame_np = (frame_np * 255).clip(0, 255).astype(np.uint8)
        
        # 转换为PIL图片
        image = Image.fromarray(frame_np, mode='RGB')
        
        # 转换为base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return f"data:image/png;base64,{img_str}"
    
    def _parse_background_color(self, color_str):
        """解析背景颜色字符串为BGR元组"""
        if color_str.startswith('#'):
            hex_color = color_str[1:]
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            return (b, g, r)
        else:
            return (0, 0, 0)
    
    def _frames_to_tensor(self, frames):
        """将帧列表转换为视频tensor"""
        if not frames:
            return torch.zeros((1, 640, 640, 3), dtype=torch.float32)
        
        frame_tensors = []
        for frame in frames:
            if isinstance(frame, np.ndarray):
                frame_tensor = torch.from_numpy(frame).float() / 255.0
                frame_tensors.append(frame_tensor)
            else:
                frame_tensors.append(frame)
        
        video_tensor = torch.stack(frame_tensors, dim=0)
        
        return video_tensor


# 注册节点
NODE_CLASS_MAPPINGS = {
    "VideoCubeTransitionNode": VideoCubeTransitionNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoCubeTransitionNode": "Video Cube Transition",
}

