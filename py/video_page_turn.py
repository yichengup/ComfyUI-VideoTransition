"""
视频翻页转场节点 - Playwright版本（批处理优化）
模拟真实书页翻动的物理效果，支持批处理和内存优化
支持左右翻页和上下翻页模式
"""

import torch
import json
import numpy as np
import time
from comfy.comfy_types.node_typing import ComfyNodeABC, InputTypeDict, IO


class VideoPageTurnNode(ComfyNodeABC):
    """视频翻页转场 - 真实书页翻动效果（批处理优化版）"""
    
    CATEGORY = "VideoTransition"
    
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "video1": (IO.IMAGE,),  # 第一个视频（当前页）
                "video2": (IO.IMAGE,),  # 第二个视频（下一页）
                "page_direction": ([
                    "right_to_left",  # 从左到右翻页
                    "left_to right",  # 从右到左翻页
                    "top_to_bottom",  # 从上到下翻页
                    "bottom_to_top",  # 从下到上翻页
                ],),
                "total_frames": (IO.INT, {"default": 60, "min": 4, "max": 300}),
                "fps": (IO.INT, {"default": 30, "min": 15, "max": 60}),
            },
            "optional": {
                "page_curl_size": (IO.FLOAT, {"default": 0.3, "min": 0.1, "max": 0.5}),
                "perspective": (IO.FLOAT, {"default": 1200.0, "min": 800.0, "max": 2000.0}),
                "use_gpu": (IO.BOOLEAN, {"default": False}),
                "batch_size": (IO.INT, {"default": 5, "min": 1, "max": 20}),
                "background_color": (IO.STRING, {"default": "#1a1a1a"}),
                "width": (IO.INT, {"default": 640, "min": 320, "max": 1920}),
                "height": (IO.INT, {"default": 640, "min": 240, "max": 1080}),
            }
        }
    
    RETURN_TYPES = (IO.IMAGE,)
    RETURN_NAMES = ("video",)
    FUNCTION = "generate_page_turn"
    
    async def generate_page_turn(
        self, 
        video1, 
        video2, 
        page_direction,
        total_frames, 
        fps,
        page_curl_size=0.3,
        perspective=1200.0,
        use_gpu=False,
        batch_size=5,
        background_color="#1a1a1a",
        width=640,
        height=640
    ):
        """生成视频翻页转场效果 - 批处理优化版本"""
        
        start_time = time.time()
        print(f"🎬 开始生成视频翻页转场（批处理优化）...")
        print(f"   翻页方向: {page_direction}")
        print(f"   翻页卷曲度: {page_curl_size}")
        print(f"   GPU加速: {'启用' if use_gpu else '禁用（SwiftShader）'}")
        print(f"   批处理大小: {batch_size}")
        
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
                print("✅ Playwright浏览器已启动（GPU硬件加速）")
            else:
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
                print("✅ Playwright浏览器已启动（SwiftShader软件渲染）")
            
            # 生成HTML模板
            html_content = self._generate_html_template(
                page_direction, page_curl_size, perspective, background_color, width, height
            )
            
            # 创建页面
            page = await browser.new_page(viewport={'width': width, 'height': height})
            
            try:
                # 加载HTML页面
                await page.set_content(html_content, wait_until='domcontentloaded', timeout=30000)
                
                # 等待页面初始化
                await page.wait_for_function("window.pageController && window.pageController.ready", timeout=5000)
                
                print("✅ Playwright初始化完成")
                
                # 批处理渲染
                print("🎬 开始批处理渲染...")
                render_start = time.time()
                
                # 使用生成器进行内存优化
                output_frames = []
                
                for batch_start in range(0, total_frames, batch_size):
                    batch_end = min(batch_start + batch_size, total_frames)
                    batch_indices = list(range(batch_start, batch_end))
                    
                    print(f"📦 处理批次 {batch_start//batch_size + 1}/{(total_frames-1)//batch_size + 1}: 帧 {batch_start+1}-{batch_end}")
                    
                    # 批处理：一次处理多个帧
                    batch_frames = await self._process_batch(
                        page, batch_indices, frame1_base64_list, frame2_base64_list, total_frames
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
        print(f"✅ 翻页转场完成，输出: {video_tensor.shape}")
        print(f"⏱️ 总耗时: {total_time:.2f}秒")
        print(f"🚀 平均每帧: {total_time/total_frames*1000:.1f}ms")
        
        return (video_tensor,)
    
    async def _process_batch(self, page, batch_indices, frame1_base64_list, frame2_base64_list, total_frames):
        """批处理渲染多个帧"""
        batch_frames = []
        
        for i in batch_indices:
            progress = i / (total_frames - 1) if total_frames > 1 else 0
            
            # 使用预计算的base64数据
            frame1_base64 = frame1_base64_list[i]
            frame2_base64 = frame2_base64_list[i]
            
            # 更新页面内容和翻页动画
            await page.evaluate(f"""
                const currentPage = document.getElementById('currentPage');
                const nextPage = document.getElementById('nextPage');
                const turningPage = document.getElementById('turningPage');
                const turningPageBack = document.getElementById('turningPageBack');
                
                if (currentPage) {{
                    currentPage.style.backgroundImage = 'url({frame1_base64})';
                }}
                
                if (nextPage) {{
                    nextPage.style.backgroundImage = 'url({frame2_base64})';
                }}
                
                if (turningPage) {{
                    turningPage.style.backgroundImage = 'url({frame1_base64})';
                }}
                
                if (turningPageBack) {{
                    turningPageBack.style.backgroundImage = 'url({frame2_base64})';
                }}
                
                // 更新翻页动画
                if (window.pageController) {{
                    window.pageController.updateTurn({progress});
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
    
    def _generate_html_template(self, page_direction, page_curl_size, perspective, background_color, width, height):
        """生成HTML模板 - 真实翻页效果（支持多方向）"""
        
        # 根据翻页方向设置CSS
        if page_direction == "right_to_left":
            # 从左到右翻页：以左边为轴心
            transform_origin = "left center"
            curl_position = "top: 0; right: 0;"
            curl_border = "border-bottom-color"
        elif page_direction == "left_to right":
            # 从右到左翻页：以右边为轴心
            transform_origin = "right center"
            curl_position = "top: 0; left: 0;"
            curl_border = "border-bottom-color"
        elif page_direction == "bottom_to_top":
            # 从下往上翻页：以顶部为轴心
            transform_origin = "top center"
            curl_position = "top: 0; left: 0;"
            curl_border = "border-right-color"
        else:
            # 从上往下翻页：以底部为轴心
            transform_origin = "bottom center"
            curl_position = "bottom: 0; left: 0;"
            curl_border = "border-right-color"
        
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
        
        .book-container {{
            width: {width}px;
            height: {height}px;
            position: relative;
            transform-style: preserve-3d;
        }}
        
        .page {{
            position: absolute;
            width: {width}px;
            height: {height}px;
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            border: none;
        }}
        
        /* 当前页（底层，不动） */
        .current-page {{
            z-index: 1;
        }}
        
        /* 下一页（底层，不动） */
        .next-page {{
            z-index: 2;
            opacity: 0;
        }}
        
        /* 翻转的页面（正面） */
        .turning-page {{
            z-index: 3;
            transform-origin: {transform_origin};
            transform-style: preserve-3d;
            backface-visibility: hidden;
            box-shadow: -5px 0 20px rgba(0,0,0,0.3);
        }}
        
        /* 翻转的页面（背面） */
        .turning-page-back {{
            z-index: 3;
            transform-origin: {transform_origin};
            transform-style: preserve-3d;
            backface-visibility: hidden;
            transform: rotateY(180deg);
        }}
        
        /* 页面阴影 */
        .page-shadow {{
            position: absolute;
            width: 100%;
            height: 100%;
            background: linear-gradient(to right, rgba(0,0,0,0), rgba(0,0,0,0.3));
            pointer-events: none;
            opacity: 0;
            z-index: 4;
        }}
        
        /* 翻页卷曲效果 */
        .page-curl {{
            position: absolute;
            {curl_position}
            width: 0;
            height: 0;
            border-style: solid;
            border-width: 0;
            border-color: transparent;
            {curl_border}: rgba(0,0,0,0.1);
            z-index: 5;
            opacity: 0;
        }}
    </style>
</head>
<body>
    <div class="book-container" id="bookContainer">
        <!-- 当前页 -->
        <div class="page current-page" id="currentPage"></div>
        
        <!-- 下一页 -->
        <div class="page next-page" id="nextPage"></div>
        
        <!-- 翻转页面的正面 -->
        <div class="page turning-page" id="turningPage"></div>
        
        <!-- 翻转页面的背面 -->
        <div class="page turning-page-back" id="turningPageBack"></div>
        
        <!-- 阴影 -->
        <div class="page-shadow" id="pageShadow"></div>
        
        <!-- 卷曲 -->
        <div class="page-curl" id="pageCurl"></div>
    </div>
    
    <script>
        class PageController {{
            constructor() {{
                this.container = document.getElementById('bookContainer');
                this.currentPage = document.getElementById('currentPage');
                this.nextPage = document.getElementById('nextPage');
                this.turningPage = document.getElementById('turningPage');
                this.turningPageBack = document.getElementById('turningPageBack');
                this.shadow = document.getElementById('pageShadow');
                this.curl = document.getElementById('pageCurl');
                this.curlSize = {page_curl_size};
                this.pageDirection = '{page_direction}';
                this.ready = true;
                
                console.log('🎬 PageController初始化完成, 翻页方向=' + this.pageDirection + ', 卷曲度=' + this.curlSize);
            }}
            
            updateTurn(progress) {{
                // progress: 0 -> 1
                
                // 缓动函数：让翻页更自然（先慢后快再慢）
                const easeProgress = this.easeInOutCubic(progress);
                
                // 翻页角度：0度 -> 180度
                const angle = easeProgress * 180;
                
                // 根据翻页方向设置变换
                let frontTransform, backTransform;
                
                switch(this.pageDirection) {{
                    case 'right_to_left':
                        // 从左到右翻页（默认）
                        frontTransform = `rotateY(${{-angle}}deg)`;
                        backTransform = `rotateY(${{180 - angle}}deg)`;
                        break;
                    
                    case 'left_to right':
                        // 从右到左翻页（以右边为轴向右翻）
                        frontTransform = `rotateY(${{angle}}deg)`;
                        backTransform = `rotateY(${{180 + angle}}deg)`;
                        break;
                    
                    case 'top_to_bottom':
                        // 从上到下翻页（顶部向下翻）
                        frontTransform = `rotateX(${{-angle}}deg)`;
                        backTransform = `rotateX(${{180 - angle}}deg)`;
                        break;
                    
                    case 'bottom_to_top':
                        // 从下往上翻页（底部向上翻）
                        frontTransform = `rotateX(${{angle}}deg)`;
                        backTransform = `rotateX(${{180 + angle}}deg)`;
                        break;
                    
                    default:
                        frontTransform = `rotateY(${{-angle}}deg)`;
                        backTransform = `rotateY(${{180 - angle}}deg)`;
                }}
                
                // 翻转的页面旋转
                this.turningPage.style.transform = frontTransform;
                this.turningPageBack.style.transform = backTransform;
                
                // 下一页逐渐显示
                this.nextPage.style.opacity = easeProgress;
                
                // 阴影效果（翻页时页面下的阴影）
                const shadowOpacity = Math.sin(easeProgress * Math.PI) * 0.5;
                this.shadow.style.opacity = shadowOpacity;
                
                // 页面卷曲效果
                this.updateCurlEffect(progress, easeProgress);
                
                // 强制重绘
                void this.container.offsetHeight;
                
                console.log(`[JS] 翻页: progress=${{progress.toFixed(2)}}, angle=${{angle.toFixed(1)}}°, direction=${{this.pageDirection}}`);
            }}
            
            updateCurlEffect(progress, easeProgress) {{
                const curlSize = this.curlSize * {width};
                
                if (progress < 0.3) {{
                    // 开始阶段：卷曲出现
                    const curlProgress = progress / 0.3;
                    this.curl.style.width = curlSize + 'px';
                    this.curl.style.height = curlSize + 'px';
                    this.curl.style.borderWidth = `0 ${{curlSize}}px ${{curlSize}}px 0`;
                    this.curl.style.opacity = curlProgress * 0.8;
                }} else if (progress > 0.7) {{
                    // 结束阶段：卷曲消失
                    const curlProgress = 1 - ((progress - 0.7) / 0.3);
                    this.curl.style.width = curlSize + 'px';
                    this.curl.style.height = curlSize + 'px';
                    this.curl.style.borderWidth = `0 ${{curlSize}}px ${{curlSize}}px 0`;
                    this.curl.style.opacity = curlProgress * 0.8;
                }} else {{
                    // 中间阶段：保持卷曲
                    this.curl.style.width = curlSize + 'px';
                    this.curl.style.height = curlSize + 'px';
                    this.curl.style.borderWidth = `0 ${{curlSize}}px ${{curlSize}}px 0`;
                    this.curl.style.opacity = 0.8;
                }}
            }}
            
            // 缓动函数：三次方缓入缓出
            easeInOutCubic(t) {{
                return t < 0.5 
                    ? 4 * t * t * t 
                    : 1 - Math.pow(-2 * t + 2, 3) / 2;
            }}
        }}
        
        window.pageController = new PageController();
        console.log('✅ PageController initialized');
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
    "VideoPageTurnNode": VideoPageTurnNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoPageTurnNode": "Video Page Turn",
}

