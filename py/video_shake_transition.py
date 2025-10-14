"""
视频抖动转场节点 - CSS3版本（剪映风格）
两个视频片段之间的抖动转场效果，模拟剪映热门转场
"""

import torch
import json
import cv2
import numpy as np
import math
import time
from comfy.comfy_types.node_typing import ComfyNodeABC, InputTypeDict, IO


class VideoShakeTransitionNode(ComfyNodeABC):
    """视频抖动转场 - 剪映风格抖动效果（批处理优化版）"""
    
    CATEGORY = "VideoTransition"
    
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "video1": (IO.IMAGE,),  # 第一个视频
                "video2": (IO.IMAGE,),  # 第二个视频
                "shake_style": ([
                    "random_shake",        # 随机抖动
                    "horizontal_shake",    # 水平抖动
                    "vertical_shake",      # 垂直抖动
                    "rotation_shake",      # 旋转抖动
                    "scale_shake",         # 缩放抖动
                    "combo_shake",         # 组合抖动
                ],),
                "total_frames": (IO.INT, {"default": 12, "min": 4, "max": 60}),
                "fps": (IO.INT, {"default": 24, "min": 15, "max": 60}),
            },
            "optional": {
                "shake_intensity": (IO.FLOAT, {"default": 20.0, "min": 5.0, "max": 50.0}),
                "shake_frequency": (IO.FLOAT, {"default": 8.0, "min": 2.0, "max": 20.0}),
                "transition_point": (IO.FLOAT, {"default": 0.5, "min": 0.2, "max": 0.8}),
                "fade_mix": (IO.BOOLEAN, {"default": True}),
                "use_gpu": (IO.BOOLEAN, {"default": False}),
                "batch_size": (IO.INT, {"default": 8, "min": 1, "max": 20}),
                "background_color": (IO.STRING, {"default": "#000000"}),
                "width": (IO.INT, {"default": 640, "min": 640, "max": 3840}),
                "height": (IO.INT, {"default": 640, "min": 360, "max": 2160}),
                "quality": (IO.INT, {"default": 90, "min": 60, "max": 100}),
            }
        }
    
    RETURN_TYPES = (IO.IMAGE,)
    RETURN_NAMES = ("video",)
    FUNCTION = "generate_shake_transition"
    
    async def generate_shake_transition(
        self, 
        video1, 
        video2,
        shake_style,
        total_frames, 
        fps,
        shake_intensity=20.0,
        shake_frequency=8.0,
        transition_point=0.5,
        fade_mix=True,
        use_gpu=False,
        batch_size=8,
        background_color="#000000",
        width=640,
        height=640,
        quality=90
    ):
        """生成视频抖动转场效果 - 剪映风格"""
        
        start_time = time.time()
        print(f"Starting shake transition: {shake_style}, {total_frames} frames")
        
        # 提取视频帧
        frames1 = self._extract_video_frames(video1)
        frames2 = self._extract_video_frames(video2)
        print(f"Video frames: {len(frames1)} -> {len(frames2)}, generating {total_frames} transition frames")
        
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
        
        # 使用Playwright直接渲染抖动效果
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
                        '--enable-gpu',
                        '--use-gl=angle',
                        '--enable-webgl',
                        '--enable-accelerated-2d-canvas',
                        '--disable-dev-shm-usage',
                        '--hide-scrollbars',
                        '--mute-audio',
                        '--no-sandbox',
                        '--disable-setuid-sandbox',
                    ]
                )
            else:
                print("Playwright browser starting with SwiftShader (CPU rendering)")
                # SwiftShader软件渲染
                browser = await playwright.chromium.launch(
                    headless=True,
                    args=[
                        '--use-angle=swiftshader',
                        '--enable-webgl',
                        '--enable-accelerated-2d-canvas',
                        '--disable-dev-shm-usage',
                        '--hide-scrollbars',
                        '--mute-audio',
                    ]
                )
            
            # 生成HTML模板
            html_content = self._generate_html_template(
                shake_style, shake_intensity, shake_frequency, transition_point,
                fade_mix, background_color, width, height
            )
            
            # 创建页面
            page = await browser.new_page(viewport={'width': width, 'height': height})
            
            try:
                # 加载HTML页面
                await page.set_content(html_content, wait_until='domcontentloaded', timeout=30000)
                
                # 等待抖动控制器初始化
                await page.wait_for_function("window.shakeController && window.shakeController.ready", timeout=10000)
                
                output_frames = []
                render_start = time.time()
                
                for batch_start in range(0, total_frames, batch_size):
                    batch_end = min(batch_start + batch_size, total_frames)
                    batch_indices = list(range(batch_start, batch_end))
                    
                    # 批处理：一次处理多个帧
                    batch_frames = await self._process_batch(
                        page, batch_indices, frame1_base64_list, frame2_base64_list, 
                        total_frames, quality
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
        print(f"Shake transition completed: {video_tensor.shape} in {total_time:.2f}s")
        
        return (video_tensor,)
    
    async def _process_batch(self, page, batch_indices, frame1_base64_list, frame2_base64_list, total_frames, quality):
        """批处理渲染多个帧"""
        batch_frames = []
        
        for i in batch_indices:
            progress = i / (total_frames - 1) if total_frames > 1 else 0
            
            # 使用预计算的base64数据
            frame1_base64 = frame1_base64_list[i]
            frame2_base64 = frame2_base64_list[i]
            
            # 更新抖动动画
            await page.evaluate(f"""
                if (window.shakeController) {{
                    window.shakeController.updateFrame({progress}, '{frame1_base64}', '{frame2_base64}');
                }}
            """)
            
            # 等待动画更新
            await page.wait_for_timeout(16)  # 抖动需要更快的更新
            
            # 强制触发重绘
            await page.evaluate("document.body.offsetHeight")
            
            # 等待渲染完成
            await page.wait_for_timeout(8)
            
            # 优化截图：使用JPEG格式
            screenshot_bytes = await page.screenshot(type='jpeg', quality=quality)
            
            # 转换为numpy数组
            from PIL import Image
            import io
            image = Image.open(io.BytesIO(screenshot_bytes)).convert('RGB')
            final_frame = np.array(image).astype(np.uint8)
            
            batch_frames.append(final_frame)
        
        return batch_frames
    
    def _generate_html_template(self, shake_style, shake_intensity, shake_frequency, transition_point, fade_mix, background_color, width, height):
        """生成HTML模板 - CSS3抖动效果（剪映风格）"""
        
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
            position: relative;
        }}
        
        .shake-container {{
            width: 100%;
            height: 100%;
            position: relative;
        }}
        
        .video-layer {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        
        .video1 {{
            z-index: 2;
        }}
        
        .video2 {{
            z-index: 1;
        }}
    </style>
</head>
<body>
    <div class="shake-container" id="shakeContainer">
        <div class="video-layer video2" id="video2Layer"></div>
        <div class="video-layer video1" id="video1Layer"></div>
    </div>
    
    <script>
        class ShakeController {{
            constructor() {{
                this.ready = false;
                this.shakeStyle = '{shake_style}';
                this.shakeIntensity = {shake_intensity};
                this.shakeFrequency = {shake_frequency};
                this.transitionPoint = {transition_point};
                this.fadeMix = {str(fade_mix).lower()};
                this.width = {width};
                this.height = {height};
                this.init();
            }}
            
            init() {{
                this.container = document.getElementById('shakeContainer');
                this.video1Layer = document.getElementById('video1Layer');
                this.video2Layer = document.getElementById('video2Layer');
                this.ready = true;
                // console.log('📳 ShakeController initialized, style=' + this.shakeStyle);
            }}
            
            updateFrame(progress, texture1Base64, texture2Base64) {{
                // 更新背景图片
                this.video1Layer.style.backgroundImage = `url(${{texture1Base64}})`;
                this.video2Layer.style.backgroundImage = `url(${{texture2Base64}})`;
                
                // 计算抖动强度（在转场点附近最强）
                const shakeAmount = this.calculateShakeAmount(progress);
                
                // 应用抖动效果
                this.applyShakeEffect(progress, shakeAmount);
                
                // 处理透明度和混合
                this.handleTransition(progress);
            }}
            
            calculateShakeAmount(progress) {{
                // 在转场点附近抖动最强，两端较弱
                const distanceFromTransition = Math.abs(progress - this.transitionPoint);
                const maxDistance = Math.max(this.transitionPoint, 1 - this.transitionPoint);
                const normalizedDistance = distanceFromTransition / maxDistance;
                
                // 使用反向二次函数，让转场点附近抖动最强
                return (1 - normalizedDistance * normalizedDistance) * this.shakeIntensity;
            }}
            
            applyShakeEffect(progress, shakeAmount) {{
                // 基于时间和频率生成抖动值
                const time = progress * 10; // 放大时间尺度
                const frequency = this.shakeFrequency;
                
                let shakeX = 0;
                let shakeY = 0;
                let shakeRotation = 0;
                let shakeScale = 1;
                
                switch(this.shakeStyle) {{
                    case 'random_shake':
                        shakeX = (Math.random() - 0.5) * shakeAmount * 2;
                        shakeY = (Math.random() - 0.5) * shakeAmount * 2;
                        shakeRotation = (Math.random() - 0.5) * shakeAmount * 0.5;
                        break;
                        
                    case 'horizontal_shake':
                        shakeX = Math.sin(time * frequency) * shakeAmount;
                        shakeY = Math.sin(time * frequency * 0.3) * shakeAmount * 0.2;
                        break;
                        
                    case 'vertical_shake':
                        shakeY = Math.sin(time * frequency) * shakeAmount;
                        shakeX = Math.sin(time * frequency * 0.3) * shakeAmount * 0.2;
                        break;
                        
                    case 'rotation_shake':
                        shakeRotation = Math.sin(time * frequency) * shakeAmount * 0.3;
                        shakeX = Math.sin(time * frequency * 0.7) * shakeAmount * 0.3;
                        shakeY = Math.sin(time * frequency * 0.5) * shakeAmount * 0.3;
                        break;
                        
                    case 'scale_shake':
                        shakeScale = 1 + Math.sin(time * frequency) * shakeAmount * 0.02;
                        shakeX = Math.sin(time * frequency * 0.8) * shakeAmount * 0.3;
                        shakeY = Math.sin(time * frequency * 0.6) * shakeAmount * 0.3;
                        break;
                        
                    case 'combo_shake':
                        // 组合多种抖动效果
                        shakeX = Math.sin(time * frequency) * shakeAmount * 0.7 + 
                                (Math.random() - 0.5) * shakeAmount * 0.3;
                        shakeY = Math.sin(time * frequency * 0.8) * shakeAmount * 0.7 + 
                                (Math.random() - 0.5) * shakeAmount * 0.3;
                        shakeRotation = Math.sin(time * frequency * 0.6) * shakeAmount * 0.2;
                        shakeScale = 1 + Math.sin(time * frequency * 1.2) * shakeAmount * 0.01;
                        break;
                }}
                
                // 应用变换到视频层
                const transform1 = `translate(${{shakeX}}px, ${{shakeY}}px) rotate(${{shakeRotation}}deg) scale(${{shakeScale}})`;
                const transform2 = `translate(${{-shakeX * 0.3}}px, ${{-shakeY * 0.3}}px) rotate(${{-shakeRotation * 0.3}}deg) scale(${{2 - shakeScale}})`;
                
                this.video1Layer.style.transform = transform1;
                this.video2Layer.style.transform = transform2;
            }}
            
            handleTransition(progress) {{
                if (this.fadeMix) {{
                    // 渐变混合模式
                    if (progress < this.transitionPoint) {{
                        // 转场前：主要显示video1
                        const fadeProgress = progress / this.transitionPoint;
                        this.video1Layer.style.opacity = '1';
                        this.video2Layer.style.opacity = (fadeProgress * 0.3).toString();
                    }} else {{
                        // 转场后：逐渐显示video2
                        const fadeProgress = (progress - this.transitionPoint) / (1 - this.transitionPoint);
                        this.video1Layer.style.opacity = (1 - fadeProgress).toString();
                        this.video2Layer.style.opacity = '1';
                    }}
                }} else {{
                    // 硬切换模式
                    if (progress < this.transitionPoint) {{
                        this.video1Layer.style.opacity = '1';
                        this.video2Layer.style.opacity = '0';
                    }} else {{
                        this.video1Layer.style.opacity = '0';
                        this.video2Layer.style.opacity = '1';
                    }}
                }}
            }}
        }}
        
        // 初始化控制器
        window.shakeController = new ShakeController();
        // console.log('✅ ShakeController initialized');
    </script>
</body>
</html>
"""
    
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
    "VideoShakeTransitionNode": VideoShakeTransitionNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoShakeTransitionNode": "Video Shake Transition",
}
