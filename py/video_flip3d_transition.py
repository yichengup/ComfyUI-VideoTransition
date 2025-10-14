"""
视频3D翻转转场节点 - Playwright版本（批处理优化）
两个视频片段之间的3D翻转转场效果，支持批处理和内存优化
"""

import torch
import json
import numpy as np
import time
from comfy.comfy_types.node_typing import ComfyNodeABC, InputTypeDict, IO


class VideoFlip3DTransitionNode(ComfyNodeABC):
    """视频3D翻转转场 - 两个视频之间的3D翻转转场（批处理优化版）"""
    
    CATEGORY = "VideoTransition"
    
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "video1": (IO.IMAGE,),  # 第一个视频（正面）
                "video2": (IO.IMAGE,),  # 第二个视频（背面）
                "transition_type": ([
                    "horizontal_flip",  # 水平翻转
                    "vertical_flip",    # 垂直翻转
                    "oblique_flip",     # 斜向翻转
                ],),
                "total_frames": (IO.INT, {"default": 60, "min": 4, "max": 300}),
                "fps": (IO.INT, {"default": 30, "min": 15, "max": 60}),
            },
            "optional": {
                "perspective": (IO.FLOAT, {"default": 1000.0, "min": 500.0, "max": 2000.0}),
                "use_gpu": (IO.BOOLEAN, {"default": False}),
                "batch_size": (IO.INT, {"default": 5, "min": 1, "max": 20}),
                "background_color": (IO.STRING, {"default": "#000000"}),
                "width": (IO.INT, {"default": 640, "min": 320, "max": 1920}),
                "height": (IO.INT, {"default": 640, "min": 240, "max": 1080}),
            }
        }
    
    RETURN_TYPES = (IO.IMAGE,)
    RETURN_NAMES = ("video",)
    FUNCTION = "generate_flip_transition"
    
    async def generate_flip_transition(
        self, 
        video1, 
        video2, 
        transition_type,
        total_frames, 
        fps,
        perspective=1000.0,
        use_gpu=False,
        batch_size=5,
        background_color="#000000",
        width=640,
        height=640
    ):
        """生成视频3D翻转转场效果 - 批处理优化版本"""
        
        start_time = time.time()
        print(f"Starting 3D flip transition: {transition_type}, {total_frames} frames")
        
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
            
            # 生成HTML模板
            html_content = self._generate_html_template(
                transition_type, perspective, background_color, width, height
            )
            
            # 创建页面
            page = await browser.new_page(viewport={'width': width, 'height': height})
            
            try:
                # 加载HTML页面
                await page.set_content(html_content, wait_until='domcontentloaded', timeout=30000)
                
                # 等待页面初始化
                await page.wait_for_function("window.flipController && window.flipController.ready", timeout=5000)
                
                # 批处理渲染
                render_start = time.time()
                
                # 使用生成器进行内存优化
                output_frames = []
                
                for batch_start in range(0, total_frames, batch_size):
                    batch_end = min(batch_start + batch_size, total_frames)
                    batch_indices = list(range(batch_start, batch_end))
                    
                    # 批处理：一次处理多个帧
                    batch_frames = await self._process_batch(
                        page, batch_indices, frame1_base64_list, frame2_base64_list, total_frames
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
        print(f"3D flip transition completed: {video_tensor.shape} in {total_time:.2f}s")
        
        return (video_tensor,)
    
    async def _process_batch(self, page, batch_indices, frame1_base64_list, frame2_base64_list, total_frames):
        """批处理渲染多个帧"""
        batch_frames = []
        
        for i in batch_indices:
            progress = i / (total_frames - 1) if total_frames > 1 else 0
            
            # 使用预计算的base64数据
            frame1_base64 = frame1_base64_list[i]
            frame2_base64 = frame2_base64_list[i]
            
            # 更新页面内容和翻转动画
            await page.evaluate(f"""
                const frontCard = document.getElementById('frontCard');
                const backCard = document.getElementById('backCard');
                
                if (frontCard) {{
                    frontCard.style.backgroundImage = 'url({frame1_base64})';
                }}
                
                if (backCard) {{
                    backCard.style.backgroundImage = 'url({frame2_base64})';
                }}
                
                // 更新翻转
                if (window.flipController) {{
                    window.flipController.updateRotation({progress});
                }}
            """)
            
            # 优化等待时间
            await page.wait_for_timeout(20)  # 从100ms减少到20ms
            
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
    
    def _generate_html_template(self, transition_type, perspective, background_color, width, height):
        """生成HTML模板 - 3D翻转卡片"""
        
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
        
        .flip-container {{
            width: {width}px;
            height: {height}px;
            position: relative;
            transform-style: preserve-3d;
        }}
        
        .flip-card {{
            position: absolute;
            width: {width}px;
            height: {height}px;
            backface-visibility: hidden;
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            border: none;
        }}
        
        .front {{
            z-index: 2;
        }}
        
        .back {{
            transform: rotateY(180deg);
            z-index: 1;
        }}
    </style>
</head>
<body>
    <div class="flip-container" id="flipContainer">
        <div class="flip-card front" id="frontCard"></div>
        <div class="flip-card back" id="backCard"></div>
    </div>
    
    <script>
        class FlipController {{
            constructor() {{
                this.container = document.getElementById('flipContainer');
                this.frontCard = document.getElementById('frontCard');
                this.backCard = document.getElementById('backCard');
                this.transitionType = '{transition_type}';
                this.ready = true;
                
                // console.log('🎬 FlipController initialized, type=' + this.transitionType);
            }}
            
            updateRotation(progress) {{
                // progress: 0 -> 1
                const angle = progress * 180;  // 0度 -> 180度
                
                let frontTransform, backTransform;
                
                switch(this.transitionType) {{
                    case 'horizontal_flip':
                        // 水平翻转（绕Y轴）
                        frontTransform = `rotateY(${{angle}}deg)`;
                        backTransform = `rotateY(${{180 + angle}}deg)`;
                        break;
                    
                    case 'vertical_flip':
                        // 垂直翻转（绕X轴）
                        frontTransform = `rotateX(${{-angle}}deg)`;
                        backTransform = `rotateX(${{180 - angle}}deg)`;
                        break;
                    
                    case 'oblique_flip':
                        // 斜向翻转 - 像书页一样从右边翻过来，带轻微抬起
                        const obliqueAngle = angle;
                        // 添加X轴偏移，模拟翻转时的轻微抬起
                        const liftAngle = Math.sin(progress * Math.PI) * 10;  // 最大抬起10度
                        frontTransform = `rotateY(${{obliqueAngle}}deg) rotateX(${{-liftAngle}}deg)`;
                        backTransform = `rotateY(${{180 + obliqueAngle}}deg) rotateX(${{-liftAngle}}deg)`;
                        break;
                    
                    default:
                        frontTransform = `rotateY(${{angle}}deg)`;
                        backTransform = `rotateY(${{180 + angle}}deg)`;
                }}
                
                this.frontCard.style.transform = frontTransform;
                this.backCard.style.transform = backTransform;
                
                // 强制重绘
                void this.container.offsetHeight;
                
                // console.log(`[JS] Update: angle=${{angle.toFixed(1)}}°, type=${{this.transitionType}}`);
            }}
        }}
        
        window.flipController = new FlipController();
        // console.log('✅ FlipController initialized');
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
    "VideoFlip3DTransitionNode": VideoFlip3DTransitionNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoFlip3DTransitionNode": "Video Flip 3D Transition",
}
