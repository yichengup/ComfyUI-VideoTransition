"""
视频爆炸转场节点 - CSS3版本（真实切割效果）
第一个视频片段分割成碎块向四周飞散，露出第二个视频片段
"""

import torch
import json
import cv2
import numpy as np
import math
import time
from comfy.comfy_types.node_typing import ComfyNodeABC, InputTypeDict, IO
from .playwright_renderer import PlaywrightRenderer


class VideoExplosionTransitionNode(ComfyNodeABC):
    """视频爆炸转场 - 碎片飞散效果（批处理优化版）"""
    
    CATEGORY = "VideoTransition"
    
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "video1": (IO.IMAGE,),  # 第一个视频（爆炸的）
                "video2": (IO.IMAGE,),  # 第二个视频（背景）
                "explosion_style": ([
                    "random_scatter",      # 随机散射
                    "radial_burst",        # 径向爆炸
                    "spiral_explosion",    # 螺旋爆炸
                    "gravity_fall",        # 重力下落
                    "wind_blow",          # 风吹效果
                    "center_burst",       # 中心爆炸
                ],),
                "total_frames": (IO.INT, {"default": 24, "min": 4, "max": 300}),
                "fps": (IO.INT, {"default": 24, "min": 15, "max": 60}),
            },
            "optional": {
                "fragment_size": (IO.INT, {"default": 64, "min": 16, "max": 128}),
                "explosion_force": (IO.FLOAT, {"default": 1.0, "min": 0.3, "max": 3.0}),
                "rotation_speed": (IO.FLOAT, {"default": 1.0, "min": 0.0, "max": 3.0}),
                "gravity_strength": (IO.FLOAT, {"default": 0.5, "min": 0.0, "max": 2.0}),
                "fade_out": (IO.BOOLEAN, {"default": True}),
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
    FUNCTION = "generate_explosion_transition"
    
    async def generate_explosion_transition(
        self, 
        video1, 
        video2,
        explosion_style,
        total_frames, 
        fps,
        fragment_size=32,
        explosion_force=1.0,
        rotation_speed=1.0,
        gravity_strength=0.5,
        fade_out=True,
        use_gpu=False,
        batch_size=5,
        background_color="#000000",
        width=640,
        height=640,
        quality=90
    ):
        """生成视频爆炸转场效果 - 碎片飞散版本"""
        
        start_time = time.time()
        print(f"💥 开始生成爆炸转场（碎片飞散效果）...")
        print(f"   爆炸样式: {explosion_style}")
        print(f"   碎片大小: {fragment_size}px")
        print(f"   爆炸力度: {explosion_force}")
        print(f"   GPU加速: {'启用' if use_gpu else '禁用（SwiftShader）'}")
        print(f"   批处理大小: {batch_size}")
        print(f"   输出尺寸: {width}x{height}")
        
        # 提取视频帧
        frames1 = self._extract_video_frames(video1)
        frames2 = self._extract_video_frames(video2)
        
        print(f"📹 视频1: {len(frames1)}帧, 视频2: {len(frames2)}帧")
        print(f"🎞️ 转场帧数: {total_frames}")
        
        # 预计算优化：提前计算所有帧的base64数据
        print("🔄 预计算帧数据...")
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
        print(f"✅ 预计算完成，耗时: {precompute_time:.2f}秒")
        
        # 使用Playwright直接渲染3D效果
        from playwright.async_api import async_playwright
        
        playwright = await async_playwright().start()
        
        try:
            # 根据GPU设置选择渲染方式
            if use_gpu:
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
                print("✅ Playwright浏览器已启动（GPU硬件加速）")
            else:
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
                print("✅ Playwright浏览器已启动（SwiftShader软件渲染）")
            
            # 生成HTML模板
            html_content = self._generate_html_template(
                explosion_style, fragment_size, explosion_force, rotation_speed,
                gravity_strength, fade_out, background_color, width, height
            )
            
            # 创建页面
            page = await browser.new_page(viewport={'width': width, 'height': height})
            
            try:
                # 加载HTML页面
                await page.set_content(html_content, wait_until='domcontentloaded', timeout=30000)
                
                # 等待爆炸控制器初始化
                await page.wait_for_function("window.explosionController && window.explosionController.ready", timeout=10000)
                
                print("✅ 爆炸场景初始化完成")
                
                # 批处理渲染
                print("🎬 开始批处理渲染...")
                render_start = time.time()
                
                output_frames = []
                
                for batch_start in range(0, total_frames, batch_size):
                    batch_end = min(batch_start + batch_size, total_frames)
                    batch_indices = list(range(batch_start, batch_end))
                    
                    print(f"📦 处理批次 {batch_start//batch_size + 1}/{(total_frames-1)//batch_size + 1}: 帧 {batch_start+1}-{batch_end}")
                    
                    # 批处理：一次处理多个帧
                    batch_frames = await self._process_batch(
                        page, batch_indices, frame1_base64_list, frame2_base64_list, 
                        total_frames, quality
                    )
                    
                    # 立即添加到结果中，避免内存积累
                    output_frames.extend(batch_frames)
                    
                    # 打印批次进度
                    batch_progress = (batch_end / total_frames) * 100
                    print(f"✅ 批次完成，总进度: {batch_progress:.1f}%")
                
                render_time = time.time() - render_start
                print(f"✅ 批处理渲染完成，耗时: {render_time:.2f}秒")
            
            finally:
                await page.close()
            
            await browser.close()
        finally:
            await playwright.stop()
        
        # 转换为tensor
        video_tensor = self._frames_to_tensor(output_frames)
        
        total_time = time.time() - start_time
        print(f"✅ 爆炸转场完成，输出: {video_tensor.shape}")
        print(f"⏱️ 总耗时: {total_time:.2f}秒")
        print(f"🚀 平均每帧: {total_time/total_frames*1000:.1f}ms")
        
        return (video_tensor,)
    
    async def _process_batch(self, page, batch_indices, frame1_base64_list, frame2_base64_list, total_frames, quality):
        """批处理渲染多个帧"""
        batch_frames = []
        
        for i in batch_indices:
            progress = i / (total_frames - 1) if total_frames > 1 else 0
            
            # 使用预计算的base64数据
            frame1_base64 = frame1_base64_list[i]
            frame2_base64 = frame2_base64_list[i]
            
            # 更新爆炸动画
            await page.evaluate(f"""
                if (window.explosionController) {{
                    window.explosionController.updateFrame({progress}, '{frame1_base64}', '{frame2_base64}');
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
    
    def _generate_html_template(self, explosion_style, fragment_size, explosion_force, rotation_speed, gravity_strength, fade_out, background_color, width, height):
        """生成HTML模板 - CSS3爆炸效果（真实切割）"""
        
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
        
        .explosion-container {{
            width: 100%;
            height: 100%;
            position: relative;
            perspective: 1000px;
        }}
        
        .background-layer {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            z-index: 1;
        }}
        
        .fragment {{
            position: absolute;
            background-size: {width}px {height}px;
            background-repeat: no-repeat;
            transform-style: preserve-3d;
            transform-origin: center;
            z-index: 2;
            border: 1px solid rgba(255,255,255,0.1);
            overflow: hidden;
        }}
    </style>
</head>
<body>
    <div class="explosion-container" id="explosionContainer">
        <div class="background-layer" id="backgroundLayer"></div>
        <!-- 碎片将由JavaScript动态生成 -->
    </div>
    
    <script>
        class ExplosionController {{
            constructor() {{
                this.ready = false;
                this.explosionStyle = '{explosion_style}';
                this.fragmentSize = {fragment_size};
                this.explosionForce = {explosion_force};
                this.rotationSpeed = {rotation_speed};
                this.gravityStrength = {gravity_strength};
                this.fadeOut = {str(fade_out).lower()};
                this.width = {width};
                this.height = {height};
                this.fragments = [];
                this.fragmentData = [];
                this.init();
            }}
            
            init() {{
                this.container = document.getElementById('explosionContainer');
                this.backgroundLayer = document.getElementById('backgroundLayer');
                this.createFragments();
                this.ready = true;
                console.log('💥 ExplosionController初始化完成, 样式=' + this.explosionStyle + ', 碎片数量=' + this.fragments.length);
            }}
            
            createFragments() {{
                // 清空容器中的碎片
                const existingFragments = this.container.querySelectorAll('.fragment');
                existingFragments.forEach(fragment => fragment.remove());
                
                this.fragments = [];
                this.fragmentData = [];
                
                const cols = Math.floor(this.width / this.fragmentSize);
                const rows = Math.floor(this.height / this.fragmentSize);
                const actualFragmentWidth = this.width / cols;
                const actualFragmentHeight = this.height / rows;
                
                for (let row = 0; row < rows; row++) {{
                    for (let col = 0; col < cols; col++) {{
                        const fragment = document.createElement('div');
                        fragment.className = 'fragment';
                        
                        // 设置碎片位置和大小
                        const x = col * actualFragmentWidth;
                        const y = row * actualFragmentHeight;
                        
                        fragment.style.left = x + 'px';
                        fragment.style.top = y + 'px';
                        fragment.style.width = actualFragmentWidth + 'px';
                        fragment.style.height = actualFragmentHeight + 'px';
                        
                        // 设置背景位置以显示对应的图片区域
                        const bgPosX = -x;
                        const bgPosY = -y;
                        fragment.style.backgroundPosition = bgPosX + 'px ' + bgPosY + 'px';
                        
                        this.container.appendChild(fragment);
                        this.fragments.push(fragment);
                        
                        // 计算碎片的运动参数
                        const fragmentData = this.calculateFragmentMotion(x, y, col, row, cols, rows);
                        this.fragmentData.push(fragmentData);
                    }}
                }}
            }}
            
            calculateFragmentMotion(x, y, col, row, cols, rows) {{
                const centerX = this.width / 2;
                const centerY = this.height / 2;
                const fragmentCenterX = x + this.fragmentSize / 2;
                const fragmentCenterY = y + this.fragmentSize / 2;
                
                let velocityX = 0;
                let velocityY = 0;
                let rotationX = 0;
                let rotationY = 0;
                let rotationZ = 0;
                
                switch(this.explosionStyle) {{
                    case 'random_scatter':
                        velocityX = (Math.random() - 0.5) * 800 * this.explosionForce;
                        velocityY = (Math.random() - 0.5) * 800 * this.explosionForce;
                        rotationX = (Math.random() - 0.5) * 720 * this.rotationSpeed;
                        rotationY = (Math.random() - 0.5) * 720 * this.rotationSpeed;
                        rotationZ = (Math.random() - 0.5) * 720 * this.rotationSpeed;
                        break;
                        
                    case 'radial_burst':
                        const angle = Math.atan2(fragmentCenterY - centerY, fragmentCenterX - centerX);
                        const distance = Math.sqrt(Math.pow(fragmentCenterX - centerX, 2) + Math.pow(fragmentCenterY - centerY, 2));
                        const force = (distance / Math.max(this.width, this.height)) * 600 * this.explosionForce;
                        velocityX = Math.cos(angle) * force;
                        velocityY = Math.sin(angle) * force;
                        rotationZ = angle * 180 / Math.PI * this.rotationSpeed;
                        break;
                        
                    case 'spiral_explosion':
                        const spiralAngle = Math.atan2(fragmentCenterY - centerY, fragmentCenterX - centerX);
                        const spiralDistance = Math.sqrt(Math.pow(fragmentCenterX - centerX, 2) + Math.pow(fragmentCenterY - centerY, 2));
                        const spiralForce = 400 * this.explosionForce;
                        const spiralOffset = (col + row) * 0.5;
                        velocityX = Math.cos(spiralAngle + spiralOffset) * spiralForce;
                        velocityY = Math.sin(spiralAngle + spiralOffset) * spiralForce;
                        rotationZ = (spiralAngle + spiralOffset) * 180 / Math.PI * this.rotationSpeed;
                        break;
                        
                    case 'gravity_fall':
                        velocityX = (Math.random() - 0.5) * 200 * this.explosionForce;
                        velocityY = -100 * this.explosionForce; // 初始向上
                        rotationX = (Math.random() - 0.5) * 360 * this.rotationSpeed;
                        break;
                        
                    case 'wind_blow':
                        velocityX = (200 + Math.random() * 300) * this.explosionForce; // 向右吹
                        velocityY = (Math.random() - 0.5) * 100 * this.explosionForce;
                        rotationZ = (Math.random() - 0.5) * 180 * this.rotationSpeed;
                        break;
                        
                    case 'center_burst':
                        const burstAngle = Math.atan2(fragmentCenterY - centerY, fragmentCenterX - centerX);
                        const burstForce = 500 * this.explosionForce;
                        velocityX = Math.cos(burstAngle) * burstForce;
                        velocityY = Math.sin(burstAngle) * burstForce;
                        rotationX = (Math.random() - 0.5) * 360 * this.rotationSpeed;
                        rotationY = (Math.random() - 0.5) * 360 * this.rotationSpeed;
                        break;
                }}
                
                return {{
                    initialX: x,
                    initialY: y,
                    velocityX: velocityX,
                    velocityY: velocityY,
                    rotationX: rotationX,
                    rotationY: rotationY,
                    rotationZ: rotationZ,
                    delay: Math.random() * 0.3 // 随机延迟0-0.3秒
                }};
            }}
            
            updateFrame(progress, texture1Base64, texture2Base64) {{
                // 更新背景层（第二个视频）
                this.backgroundLayer.style.backgroundImage = `url(${{texture2Base64}})`;
                
                // 更新所有碎片
                this.fragments.forEach((fragment, index) => {{
                    this.updateFragment(fragment, index, progress, texture1Base64);
                }});
            }}
            
            updateFragment(fragment, index, progress, texture1Base64) {{
                const data = this.fragmentData[index];
                
                // 考虑延迟的进度
                const adjustedProgress = Math.max(0, Math.min(1, (progress - data.delay) / (1 - data.delay)));
                
                // 使用缓动函数
                const easeProgress = this.easeOutCubic(adjustedProgress);
                
                if (adjustedProgress <= 0) {{
                    // 还没开始爆炸，显示原始位置
                    fragment.style.backgroundImage = `url(${{texture1Base64}})`;
                    fragment.style.transform = 'translate(0, 0) rotateX(0deg) rotateY(0deg) rotateZ(0deg)';
                    fragment.style.opacity = '1';
                    fragment.style.display = 'block';
                }} else if (adjustedProgress >= 1) {{
                    // 爆炸完成，隐藏碎片
                    fragment.style.display = 'none';
                }} else {{
                    // 爆炸过程中
                    fragment.style.backgroundImage = `url(${{texture1Base64}})`;
                    fragment.style.display = 'block';
                    
                    // 计算位置
                    let currentX = data.initialX + data.velocityX * easeProgress;
                    let currentY = data.initialY + data.velocityY * easeProgress;
                    
                    // 重力效果（仅对gravity_fall样式）
                    if (this.explosionStyle === 'gravity_fall') {{
                        currentY += 0.5 * this.gravityStrength * 500 * easeProgress * easeProgress;
                    }}
                    
                    // 计算旋转
                    const rotX = data.rotationX * easeProgress;
                    const rotY = data.rotationY * easeProgress;
                    const rotZ = data.rotationZ * easeProgress;
                    
                    // 应用变换
                    const translateX = currentX - data.initialX;
                    const translateY = currentY - data.initialY;
                    fragment.style.transform = `translate(${{translateX}}px, ${{translateY}}px) rotateX(${{rotX}}deg) rotateY(${{rotY}}deg) rotateZ(${{rotZ}}deg)`;
                    
                    // 淡出效果
                    if (this.fadeOut) {{
                        const opacity = 1 - easeProgress * 0.8; // 保留一些透明度直到最后
                        fragment.style.opacity = Math.max(0, opacity).toString();
                    }} else {{
                        fragment.style.opacity = '1';
                    }}
                }}
            }}
            
            // 缓动函数
            easeOutCubic(t) {{
                return 1 - Math.pow(1 - t, 3);
            }}
        }}
        
        // 初始化控制器
        window.explosionController = new ExplosionController();
        console.log('✅ ExplosionController initialized');
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
    "VideoExplosionTransitionNode": VideoExplosionTransitionNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoExplosionTransitionNode": "Video Explosion Transition",
}
