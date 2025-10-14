"""
视频百叶窗转场节点 - CSS3版本（批处理优化）
两个视频片段之间的百叶窗转场效果，支持批处理和GPU加速
"""

import torch
import json
import cv2
import numpy as np
import math
import time
from comfy.comfy_types.node_typing import ComfyNodeABC, InputTypeDict, IO


class VideoBlindsTransitionNode(ComfyNodeABC):
    """视频百叶窗转场 - 专业百叶窗翻转效果（批处理优化版）"""
    
    CATEGORY = "VideoTransition"
    
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "video1": (IO.IMAGE,),  # 第一个视频
                "video2": (IO.IMAGE,),  # 第二个视频
                "blinds_direction": ([
                    "horizontal",        # 水平百叶窗
                    "vertical",         # 垂直百叶窗
                    "diagonal_right",   # 右对角百叶窗
                    "diagonal_left",    # 左对角百叶窗
                ],),
                "total_frames": (IO.INT, {"default": 24, "min": 4, "max": 300}),
                "fps": (IO.INT, {"default": 24, "min": 15, "max": 60}),
            },
            "optional": {
                "blinds_count": (IO.INT, {"default": 4, "min": 4, "max": 30}),
                "flip_duration": (IO.FLOAT, {"default": 0.6, "min": 0.3, "max": 1.0}),
                "stagger_delay": (IO.FLOAT, {"default": 0.05, "min": 0.0, "max": 0.2}),
                "use_gpu": (IO.BOOLEAN, {"default": False}),
                "batch_size": (IO.INT, {"default": 5, "min": 1, "max": 20}),
                "background_color": (IO.STRING, {"default": "#000000"}),
                "width": (IO.INT, {"default": 640, "min": 640, "max": 3840}),
                "height": (IO.INT, {"default": 640, "min": 360, "max": 2160}),
                "quality": (IO.INT, {"default": 90, "min": 60, "max": 100}),
            }
        }
    
    RETURN_TYPES = (IO.IMAGE,)
    RETURN_NAMES = ("video",)
    FUNCTION = "generate_blinds_transition"
    
    async def generate_blinds_transition(
        self, 
        video1, 
        video2,
        blinds_direction,
        total_frames, 
        fps,
        blinds_count=12,
        flip_duration=0.6,
        stagger_delay=0.05,
        use_gpu=False,
        batch_size=5,
        background_color="#000000",
        width=1920,
        height=1080,
        quality=90
    ):
        """生成视频百叶窗转场效果 - 批处理优化版本"""
        
        start_time = time.time()
        print(f"Starting blinds transition: {blinds_direction}, {total_frames} frames")
        
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
                blinds_direction, blinds_count, flip_duration, stagger_delay,
                background_color, width, height
            )
            
            # 创建页面
            page = await browser.new_page(viewport={'width': width, 'height': height})
            
            try:
                # 加载HTML页面
                await page.set_content(html_content, wait_until='domcontentloaded', timeout=30000)
                
                # 等待百叶窗控制器初始化
                await page.wait_for_function("window.blindsController && window.blindsController.ready", timeout=10000)
                
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
        print(f"Blinds transition completed: {video_tensor.shape} in {total_time:.2f}s")
        
        return (video_tensor,)
    
    async def _process_batch(self, page, batch_indices, frame1_base64_list, frame2_base64_list, total_frames, quality):
        """批处理渲染多个帧"""
        batch_frames = []
        
        for i in batch_indices:
            progress = i / (total_frames - 1) if total_frames > 1 else 0
            
            # 使用预计算的base64数据
            frame1_base64 = frame1_base64_list[i]
            frame2_base64 = frame2_base64_list[i]
            
            # 更新百叶窗动画
            await page.evaluate(f"""
                if (window.blindsController) {{
                    window.blindsController.updateFrame({progress}, '{frame1_base64}', '{frame2_base64}');
                }}
            """)
            
            # 等待动画更新
            await page.wait_for_timeout(25)
            
            # 强制触发重绘
            await page.evaluate("document.body.offsetHeight")
            
            # 等待渲染完成
            await page.wait_for_timeout(15)
            
            # 优化截图：使用JPEG格式
            screenshot_bytes = await page.screenshot(type='jpeg', quality=quality)
            
            # 转换为numpy数组
            from PIL import Image
            import io
            image = Image.open(io.BytesIO(screenshot_bytes)).convert('RGB')
            final_frame = np.array(image).astype(np.uint8)
            
            batch_frames.append(final_frame)
        
        return batch_frames
    
    def _generate_html_template(self, blinds_direction, blinds_count, flip_duration, stagger_delay, background_color, width, height):
        """生成HTML模板 - CSS3百叶窗效果"""
        
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
        
        .blinds-container {{
            width: 100%;
            height: 100%;
            position: relative;
            perspective: 1000px;
        }}
        
        .blind-strip {{
            position: absolute;
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            transform-style: preserve-3d;
            transform-origin: center;
            transition: transform 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        }}
        
        .blind-strip.horizontal {{
            width: 100%;
            left: 0;
        }}
        
        .blind-strip.vertical {{
            height: 100%;
            top: 0;
        }}
        
        .blind-strip.diagonal-right {{
            transform-origin: top left;
        }}
        
        .blind-strip.diagonal-left {{
            transform-origin: top right;
        }}
    </style>
</head>
<body>
    <div class="blinds-container" id="blindsContainer">
        <!-- 百叶窗条带将由JavaScript动态生成 -->
    </div>
    
    <script>
        class BlindsController {{
            constructor() {{
                this.ready = false;
                this.direction = '{blinds_direction}';
                this.blindsCount = {blinds_count};
                this.flipDuration = {flip_duration};
                this.staggerDelay = {stagger_delay};
                this.width = {width};
                this.height = {height};
                this.blinds = [];
                this.init();
            }}
            
            init() {{
                this.container = document.getElementById('blindsContainer');
                this.createBlinds();
                this.ready = true;
                // console.log('🎬 BlindsController initialized, direction=' + this.direction + ', count=' + this.blindsCount);
            }}
            
            createBlinds() {{
                // 清空容器
                this.container.innerHTML = '';
                this.blinds = [];
                
                for (let i = 0; i < this.blindsCount; i++) {{
                    const blind = document.createElement('div');
                    blind.className = `blind-strip ${{this.direction}}`;
                    
                    // 根据方向设置百叶窗条带的位置和大小
                    this.setupBlindStrip(blind, i);
                    
                    this.container.appendChild(blind);
                    this.blinds.push(blind);
                }}
            }}
            
            setupBlindStrip(blind, index) {{
                const stripSize = this.getStripSize();
                const position = this.getStripPosition(index, stripSize);
                
                switch(this.direction) {{
                    case 'horizontal':
                        blind.style.height = stripSize + 'px';
                        blind.style.top = position + 'px';
                        break;
                    case 'vertical':
                        blind.style.width = stripSize + 'px';
                        blind.style.left = position + 'px';
                        break;
                    case 'diagonal_right':
                    case 'diagonal_left':
                        blind.style.width = stripSize + 'px';
                        blind.style.height = this.height + 'px';
                        blind.style.left = position + 'px';
                        blind.style.top = '0px';
                        break;
                }}
            }}
            
            getStripSize() {{
                switch(this.direction) {{
                    case 'horizontal':
                        return Math.floor(this.height / this.blindsCount);
                    case 'vertical':
                    case 'diagonal_right':
                    case 'diagonal_left':
                        return Math.floor(this.width / this.blindsCount);
                    default:
                        return 50;
                }}
            }}
            
            getStripPosition(index, stripSize) {{
                return index * stripSize;
            }}
            
            updateFrame(progress, texture1Base64, texture2Base64) {{
                // 更新所有百叶窗条带
                this.blinds.forEach((blind, index) => {{
                    this.updateBlindStrip(blind, index, progress, texture1Base64, texture2Base64);
                }});
            }}
            
            updateBlindStrip(blind, index, progress, texture1Base64, texture2Base64) {{
                // 计算每个条带的翻转时机（错开动画）
                const stripDelay = index * this.staggerDelay;
                const adjustedProgress = Math.max(0, Math.min(1, (progress - stripDelay) / this.flipDuration));
                
                // 使用缓动函数让翻转更自然
                const easeProgress = this.easeInOutCubic(adjustedProgress);
                
                // 计算当前条带对应的图片切割区域
                const textureMapping = this.getTextureMapping(index);
                
                if (adjustedProgress <= 0) {{
                    // 还没开始翻转，显示第一个视频的对应切片
                    blind.style.backgroundImage = `url(${{texture1Base64}})`;
                    blind.style.backgroundPosition = textureMapping.position;
                    blind.style.backgroundSize = textureMapping.size;
                    blind.style.transform = this.getTransform(0);
                }} else if (adjustedProgress >= 1) {{
                    // 翻转完成，显示第二个视频的对应切片
                    blind.style.backgroundImage = `url(${{texture2Base64}})`;
                    blind.style.backgroundPosition = textureMapping.position;
                    blind.style.backgroundSize = textureMapping.size;
                    blind.style.transform = this.getTransform(0);
                }} else {{
                    // 翻转过程中
                    if (easeProgress < 0.5) {{
                        // 前半段：显示第一个视频的切片，开始翻转
                        blind.style.backgroundImage = `url(${{texture1Base64}})`;
                        blind.style.backgroundPosition = textureMapping.position;
                        blind.style.backgroundSize = textureMapping.size;
                        blind.style.transform = this.getTransform(easeProgress * 2);
                    }} else {{
                        // 后半段：显示第二个视频的切片，完成翻转
                        blind.style.backgroundImage = `url(${{texture2Base64}})`;
                        blind.style.backgroundPosition = textureMapping.position;
                        blind.style.backgroundSize = textureMapping.size;
                        blind.style.transform = this.getTransform(2 - easeProgress * 2);
                    }}
                }}
            }}
            
            getTextureMapping(index) {{
                // 计算每个百叶窗条带对应的图片切割区域
                const stripSize = this.getStripSize();
                
                switch(this.direction) {{
                    case 'horizontal':
                        // 水平百叶窗：每个条带显示图片的水平切片
                        const yOffset = index * stripSize;
                        return {{
                            position: `0px -${{yOffset}}px`,
                            size: `${{this.width}}px ${{this.height}}px`
                        }};
                        
                    case 'vertical':
                        // 垂直百叶窗：每个条带显示图片的垂直切片
                        const xOffset = index * stripSize;
                        return {{
                            position: `-${{xOffset}}px 0px`,
                            size: `${{this.width}}px ${{this.height}}px`
                        }};
                        
                    case 'diagonal_right':
                    case 'diagonal_left':
                        // 对角百叶窗：每个条带显示图片的垂直切片（类似垂直）
                        const xOffsetDiag = index * stripSize;
                        return {{
                            position: `-${{xOffsetDiag}}px 0px`,
                            size: `${{this.width}}px ${{this.height}}px`
                        }};
                        
                    default:
                        return {{
                            position: '0px 0px',
                            size: `${{this.width}}px ${{this.height}}px`
                        }};
                }}
            }}
            
            getTransform(flipProgress) {{
                // flipProgress: 0 = 正面, 1 = 完全翻转
                const angle = flipProgress * 180;
                
                switch(this.direction) {{
                    case 'horizontal':
                        return `rotateX(${{angle}}deg)`;
                    case 'vertical':
                        return `rotateY(${{angle}}deg)`;
                    case 'diagonal_right':
                        return `rotateY(${{angle}}deg) rotateX(${{angle * 0.3}}deg)`;
                    case 'diagonal_left':
                        return `rotateY(${{-angle}}deg) rotateX(${{angle * 0.3}}deg)`;
                    default:
                        return `rotateY(${{angle}}deg)`;
                }}
            }}
            
            // 缓动函数
            easeInOutCubic(t) {{
                return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
            }}
        }}
        
        // 初始化控制器
        window.blindsController = new BlindsController();
        // console.log('✅ BlindsController initialized');
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
            return torch.zeros((1, 1080, 1920, 3), dtype=torch.float32)
        
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
    "VideoBlindsTransitionNode": VideoBlindsTransitionNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoBlindsTransitionNode": "Video Blinds Transition",
}
