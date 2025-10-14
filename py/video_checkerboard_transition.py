"""
视频棋盘格转场节点 - CSS3版本（真实切割效果）
两个视频片段之间的棋盘格转场效果，每个方格显示对应的图片区域
"""

import torch
import json
import cv2
import numpy as np
import math
import time
from comfy.comfy_types.node_typing import ComfyNodeABC, InputTypeDict, IO


class VideoCheckerboardTransitionNode(ComfyNodeABC):
    """视频棋盘格转场 - 真实图片切割效果（批处理优化版）"""
    
    CATEGORY = "VideoTransition"
    
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "video1": (IO.IMAGE,),  # 第一个视频
                "video2": (IO.IMAGE,),  # 第二个视频
                "transition_style": ([
                    "flip_squares",        # 方格翻转
                    "scale_squares",       # 方格缩放
                    "rotate_squares",      # 方格旋转
                    "slide_squares",       # 方格滑动
                    "wave_squares",        # 波浪式方格
                ],),
                "total_frames": (IO.INT, {"default": 24, "min": 4, "max": 300}),
                "fps": (IO.INT, {"default": 24, "min": 15, "max": 60}),
            },
            "optional": {
                "grid_size": (IO.INT, {"default": 8, "min": 4, "max": 20}),
                "animation_duration": (IO.FLOAT, {"default": 0.8, "min": 0.3, "max": 1.5}),
                "stagger_delay": (IO.FLOAT, {"default": 0.02, "min": 0.0, "max": 0.1}),
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
    FUNCTION = "generate_checkerboard_transition"
    
    async def generate_checkerboard_transition(
        self, 
        video1, 
        video2,
        transition_style,
        total_frames, 
        fps,
        grid_size=8,
        animation_duration=0.8,
        stagger_delay=0.02,
        use_gpu=False,
        batch_size=5,
        background_color="#000000",
        width=640,
        height=640,
        quality=90
    ):
        """生成视频棋盘格转场效果 - 真实切割版本"""
        
        start_time = time.time()
        print(f"Starting checkerboard transition: {transition_style}, {total_frames} frames")
        
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
                transition_style, grid_size, animation_duration, stagger_delay,
                background_color, width, height
            )
            
            # 创建页面
            page = await browser.new_page(viewport={'width': width, 'height': height})
            
            try:
                # 加载HTML页面
                await page.set_content(html_content, wait_until='domcontentloaded', timeout=30000)
                
                # 等待棋盘格控制器初始化
                await page.wait_for_function("window.checkerboardController && window.checkerboardController.ready", timeout=10000)
                
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
        print(f"Checkerboard transition completed: {video_tensor.shape} in {total_time:.2f}s")
        
        return (video_tensor,)
    
    async def _process_batch(self, page, batch_indices, frame1_base64_list, frame2_base64_list, total_frames, quality):
        """批处理渲染多个帧"""
        batch_frames = []
        
        for i in batch_indices:
            progress = i / (total_frames - 1) if total_frames > 1 else 0
            
            # 使用预计算的base64数据
            frame1_base64 = frame1_base64_list[i]
            frame2_base64 = frame2_base64_list[i]
            
            # 更新棋盘格动画
            await page.evaluate(f"""
                if (window.checkerboardController) {{
                    window.checkerboardController.updateFrame({progress}, '{frame1_base64}', '{frame2_base64}');
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
    
    def _generate_html_template(self, transition_style, grid_size, animation_duration, stagger_delay, background_color, width, height):
        """生成HTML模板 - CSS3棋盘格效果（真实切割）"""
        
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
        
        .checkerboard-container {{
            width: 100%;
            height: 100%;
            position: relative;
            perspective: 1000px;
        }}
        
        .checker-square {{
            position: absolute;
            background-size: {width}px {height}px;
            background-repeat: no-repeat;
            transform-style: preserve-3d;
            transform-origin: center;
            transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);
            border: 1px solid rgba(255,255,255,0.05);
            overflow: hidden;
        }}
        
        .checker-square.flip {{
            backface-visibility: hidden;
        }}
        
        .checker-square.scale {{
            transform: scale(1);
        }}
        
        .checker-square.rotate {{
            transform: rotate(0deg);
        }}
        
        .checker-square.slide {{
            transform: translateX(0) translateY(0);
        }}
        
        .checker-square.wave {{
            transform: translateZ(0) rotateX(0deg) rotateY(0deg);
        }}
    </style>
</head>
<body>
    <div class="checkerboard-container" id="checkerboardContainer">
        <!-- 棋盘格方块将由JavaScript动态生成 -->
    </div>
    
    <script>
        class CheckerboardController {{
            constructor() {{
                this.ready = false;
                this.transitionStyle = '{transition_style}';
                this.gridSize = {grid_size};
                this.animationDuration = {animation_duration};
                this.staggerDelay = {stagger_delay};
                this.width = {width};
                this.height = {height};
                this.squares = [];
                this.init();
            }}
            
            init() {{
                this.container = document.getElementById('checkerboardContainer');
                this.createCheckerboard();
                this.ready = true;
                // console.log('🎬 CheckerboardController initialized, style=' + this.transitionStyle + ', grid=' + this.gridSize + 'x' + this.gridSize);
            }}
            
            createCheckerboard() {{
                // 清空容器
                this.container.innerHTML = '';
                this.squares = [];
                
                const squareWidth = Math.floor(this.width / this.gridSize);
                const squareHeight = Math.floor(this.height / this.gridSize);
                
                for (let row = 0; row < this.gridSize; row++) {{
                    for (let col = 0; col < this.gridSize; col++) {{
                        const square = document.createElement('div');
                        square.className = `checker-square ${{this.transitionStyle.split('_')[1]}}`;
                        
                        // 设置方块位置和大小
                        square.style.left = (col * squareWidth) + 'px';
                        square.style.top = (row * squareHeight) + 'px';
                        square.style.width = squareWidth + 'px';
                        square.style.height = squareHeight + 'px';
                        
                        // 关键：设置背景位置以显示对应的图片区域（真实切割效果）
                        const bgPosX = -(col * squareWidth);
                        const bgPosY = -(row * squareHeight);
                        square.style.backgroundPosition = bgPosX + 'px ' + bgPosY + 'px';
                        
                        // 存储方块信息
                        square.dataset.row = row;
                        square.dataset.col = col;
                        square.dataset.index = row * this.gridSize + col;
                        square.dataset.bgPosX = bgPosX;
                        square.dataset.bgPosY = bgPosY;
                        
                        this.container.appendChild(square);
                        this.squares.push(square);
                    }}
                }}
            }}
            
            updateFrame(progress, texture1Base64, texture2Base64) {{
                // 更新所有棋盘格方块
                this.squares.forEach((square, index) => {{
                    this.updateSquare(square, index, progress, texture1Base64, texture2Base64);
                }});
            }}
            
            updateSquare(square, index, progress, texture1Base64, texture2Base64) {{
                const row = parseInt(square.dataset.row);
                const col = parseInt(square.dataset.col);
                const bgPosX = parseInt(square.dataset.bgPosX);
                const bgPosY = parseInt(square.dataset.bgPosY);
                
                // 计算每个方块的动画延迟（基于不同的模式）
                const delay = this.calculateDelay(row, col, index);
                const adjustedProgress = Math.max(0, Math.min(1, (progress - delay) / this.animationDuration));
                
                // 使用缓动函数让动画更自然
                const easeProgress = this.easeInOutCubic(adjustedProgress);
                
                // 根据转场样式应用不同的动画
                switch(this.transitionStyle) {{
                    case 'flip_squares':
                        this.animateFlip(square, easeProgress, texture1Base64, texture2Base64, bgPosX, bgPosY);
                        break;
                    case 'scale_squares':
                        this.animateScale(square, easeProgress, texture1Base64, texture2Base64, bgPosX, bgPosY);
                        break;
                    case 'rotate_squares':
                        this.animateRotate(square, easeProgress, texture1Base64, texture2Base64, bgPosX, bgPosY);
                        break;
                    case 'slide_squares':
                        this.animateSlide(square, easeProgress, texture1Base64, texture2Base64, bgPosX, bgPosY, row, col);
                        break;
                    case 'wave_squares':
                        this.animateWave(square, easeProgress, texture1Base64, texture2Base64, bgPosX, bgPosY, row, col);
                        break;
                    default:
                        this.animateFlip(square, easeProgress, texture1Base64, texture2Base64, bgPosX, bgPosY);
                }}
            }}
            
            calculateDelay(row, col, index) {{
                switch(this.transitionStyle) {{
                    case 'flip_squares':
                        // 对角线延迟
                        return (row + col) * this.staggerDelay;
                    case 'scale_squares':
                        // 从中心向外
                        const centerRow = this.gridSize / 2;
                        const centerCol = this.gridSize / 2;
                        const distance = Math.sqrt(Math.pow(row - centerRow, 2) + Math.pow(col - centerCol, 2));
                        return distance * this.staggerDelay;
                    case 'rotate_squares':
                        // 螺旋延迟
                        return this.getSpiralDelay(row, col) * this.staggerDelay;
                    case 'slide_squares':
                        // 从左到右
                        return col * this.staggerDelay;
                    case 'wave_squares':
                        // 波浪延迟
                        return Math.sin(col * Math.PI / this.gridSize) * this.staggerDelay;
                    default:
                        return index * this.staggerDelay;
                }}
            }}
            
            getSpiralDelay(row, col) {{
                // 简化的螺旋计算
                const centerRow = Math.floor(this.gridSize / 2);
                const centerCol = Math.floor(this.gridSize / 2);
                const angle = Math.atan2(row - centerRow, col - centerCol);
                const distance = Math.sqrt(Math.pow(row - centerRow, 2) + Math.pow(col - centerCol, 2));
                return angle + distance;
            }}
            
            animateFlip(square, progress, texture1Base64, texture2Base64, bgPosX, bgPosY) {{
                const angle = progress * 180;
                
                if (progress <= 0) {{
                    square.style.backgroundImage = `url(${{texture1Base64}})`;
                    square.style.backgroundPosition = bgPosX + 'px ' + bgPosY + 'px';
                    square.style.transform = 'rotateY(0deg)';
                }} else if (progress >= 1) {{
                    square.style.backgroundImage = `url(${{texture2Base64}})`;
                    square.style.backgroundPosition = bgPosX + 'px ' + bgPosY + 'px';
                    square.style.transform = 'rotateY(0deg)';
                }} else {{
                    if (progress < 0.5) {{
                        square.style.backgroundImage = `url(${{texture1Base64}})`;
                        square.style.backgroundPosition = bgPosX + 'px ' + bgPosY + 'px';
                        square.style.transform = `rotateY(${{angle}}deg)`;
                    }} else {{
                        square.style.backgroundImage = `url(${{texture2Base64}})`;
                        square.style.backgroundPosition = bgPosX + 'px ' + bgPosY + 'px';
                        square.style.transform = `rotateY(${{180 - angle}}deg)`;
                    }}
                }}
            }}
            
            animateScale(square, progress, texture1Base64, texture2Base64, bgPosX, bgPosY) {{
                if (progress <= 0) {{
                    square.style.backgroundImage = `url(${{texture1Base64}})`;
                    square.style.backgroundPosition = bgPosX + 'px ' + bgPosY + 'px';
                    square.style.transform = 'scale(1)';
                }} else if (progress >= 1) {{
                    square.style.backgroundImage = `url(${{texture2Base64}})`;
                    square.style.backgroundPosition = bgPosX + 'px ' + bgPosY + 'px';
                    square.style.transform = 'scale(1)';
                }} else {{
                    if (progress < 0.5) {{
                        square.style.backgroundImage = `url(${{texture1Base64}})`;
                        square.style.backgroundPosition = bgPosX + 'px ' + bgPosY + 'px';
                        const scale = 1 - progress;
                        square.style.transform = `scale(${{scale}})`;
                    }} else {{
                        square.style.backgroundImage = `url(${{texture2Base64}})`;
                        square.style.backgroundPosition = bgPosX + 'px ' + bgPosY + 'px';
                        const scale = progress;
                        square.style.transform = `scale(${{scale}})`;
                    }}
                }}
            }}
            
            animateRotate(square, progress, texture1Base64, texture2Base64, bgPosX, bgPosY) {{
                const angle = progress * 360;
                
                if (progress <= 0) {{
                    square.style.backgroundImage = `url(${{texture1Base64}})`;
                    square.style.backgroundPosition = bgPosX + 'px ' + bgPosY + 'px';
                    square.style.transform = 'rotate(0deg)';
                }} else if (progress >= 1) {{
                    square.style.backgroundImage = `url(${{texture2Base64}})`;
                    square.style.backgroundPosition = bgPosX + 'px ' + bgPosY + 'px';
                    square.style.transform = 'rotate(0deg)';
                }} else {{
                    if (progress < 0.5) {{
                        square.style.backgroundImage = `url(${{texture1Base64}})`;
                        square.style.backgroundPosition = bgPosX + 'px ' + bgPosY + 'px';
                    }} else {{
                        square.style.backgroundImage = `url(${{texture2Base64}})`;
                        square.style.backgroundPosition = bgPosX + 'px ' + bgPosY + 'px';
                    }}
                    square.style.transform = `rotate(${{angle}}deg)`;
                }}
            }}
            
            animateSlide(square, progress, texture1Base64, texture2Base64, bgPosX, bgPosY, row, col) {{
                const slideDistance = 100; // 滑动距离（像素）
                
                if (progress <= 0) {{
                    square.style.backgroundImage = `url(${{texture1Base64}})`;
                    square.style.backgroundPosition = bgPosX + 'px ' + bgPosY + 'px';
                    square.style.transform = 'translateX(0)';
                }} else if (progress >= 1) {{
                    square.style.backgroundImage = `url(${{texture2Base64}})`;
                    square.style.backgroundPosition = bgPosX + 'px ' + bgPosY + 'px';
                    square.style.transform = 'translateX(0)';
                }} else {{
                    if (progress < 0.5) {{
                        square.style.backgroundImage = `url(${{texture1Base64}})`;
                        square.style.backgroundPosition = bgPosX + 'px ' + bgPosY + 'px';
                        const offset = -progress * slideDistance * 2;
                        square.style.transform = `translateX(${{offset}}px)`;
                    }} else {{
                        square.style.backgroundImage = `url(${{texture2Base64}})`;
                        square.style.backgroundPosition = bgPosX + 'px ' + bgPosY + 'px';
                        const offset = (1 - progress) * slideDistance * 2;
                        square.style.transform = `translateX(${{offset}}px)`;
                    }}
                }}
            }}
            
            animateWave(square, progress, texture1Base64, texture2Base64, bgPosX, bgPosY, row, col) {{
                const waveHeight = 50; // 波浪高度
                const waveFreq = 2; // 波浪频率
                
                if (progress <= 0) {{
                    square.style.backgroundImage = `url(${{texture1Base64}})`;
                    square.style.backgroundPosition = bgPosX + 'px ' + bgPosY + 'px';
                    square.style.transform = 'translateZ(0) rotateX(0deg)';
                }} else if (progress >= 1) {{
                    square.style.backgroundImage = `url(${{texture2Base64}})`;
                    square.style.backgroundPosition = bgPosX + 'px ' + bgPosY + 'px';
                    square.style.transform = 'translateZ(0) rotateX(0deg)';
                }} else {{
                    if (progress < 0.5) {{
                        square.style.backgroundImage = `url(${{texture1Base64}})`;
                        square.style.backgroundPosition = bgPosX + 'px ' + bgPosY + 'px';
                    }} else {{
                        square.style.backgroundImage = `url(${{texture2Base64}})`;
                        square.style.backgroundPosition = bgPosX + 'px ' + bgPosY + 'px';
                    }}
                    
                    const waveOffset = Math.sin(progress * Math.PI * waveFreq + col * 0.5) * waveHeight;
                    const rotateX = Math.sin(progress * Math.PI + col * 0.3) * 15;
                    square.style.transform = `translateZ(${{waveOffset}}px) rotateX(${{rotateX}}deg)`;
                }}
            }}
            
            // 缓动函数
            easeInOutCubic(t) {{
                return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
            }}
        }}
        
        // 初始化控制器
        window.checkerboardController = new CheckerboardController();
        // console.log('✅ CheckerboardController initialized');
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
    "VideoCheckerboardTransitionNode": VideoCheckerboardTransitionNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoCheckerboardTransitionNode": "Video Checkerboard Transition",
}