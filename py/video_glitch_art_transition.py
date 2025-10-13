"""
视频故障艺术转场节点 - WebGL版本（赛博朋克故障效果）
数字故障、像素错位、RGB分离、信号干扰等效果
"""

import torch
import json
import cv2
import numpy as np
import math
import time
from comfy.comfy_types.node_typing import ComfyNodeABC, InputTypeDict, IO
from .playwright_renderer import PlaywrightRenderer


class VideoGlitchArtTransitionNode(ComfyNodeABC):
    """视频故障艺术转场 - 赛博朋克数字故障效果（批处理优化版）"""
    
    CATEGORY = "VideoTransition"
    
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "video1": (IO.IMAGE,),  # 第一个视频（故障源）
                "video2": (IO.IMAGE,),  # 第二个视频（目标）
                "glitch_style": ([
                    "signal_interference", # 信号干扰
                    "data_moshing",        # 数据混合
                    "pixel_sorting",       # 像素排序
                    "matrix_rain",         # 矩阵雨
                ],),
                "total_frames": (IO.INT, {"default": 36, "min": 4, "max": 120}),
                "fps": (IO.INT, {"default": 24, "min": 15, "max": 60}),
            },
            "optional": {
                "glitch_intensity": (IO.FLOAT, {"default": 1.0, "min": 0.1, "max": 3.0}),
                "corruption_rate": (IO.FLOAT, {"default": 0.3, "min": 0.1, "max": 0.8}),
                "scanline_frequency": (IO.FLOAT, {"default": 5.0, "min": 1.0, "max": 20.0}),
                "noise_amount": (IO.FLOAT, {"default": 0.2, "min": 0.0, "max": 0.5}),
                "digital_artifacts": (IO.BOOLEAN, {"default": True}),
                "color_bleeding": (IO.BOOLEAN, {"default": True}),
                "screen_tear": (IO.BOOLEAN, {"default": True}),
                "use_gpu": (IO.BOOLEAN, {"default": True}),
                "batch_size": (IO.INT, {"default": 6, "min": 1, "max": 15}),
                "background_color": (IO.STRING, {"default": "#000000"}),
                "width": (IO.INT, {"default": 640, "min": 640, "max": 3840}),
                "height": (IO.INT, {"default": 640, "min": 360, "max": 2160}),
                "quality": (IO.INT, {"default": 90, "min": 60, "max": 100}),
            }
        }
    
    RETURN_TYPES = (IO.IMAGE,)
    RETURN_NAMES = ("video",)
    FUNCTION = "generate_glitch_art_transition"
    
    async def generate_glitch_art_transition(
        self, 
        video1, 
        video2,
        glitch_style,
        total_frames, 
        fps,
        glitch_intensity=1.0,
        corruption_rate=0.3,
        scanline_frequency=5.0,
        noise_amount=0.2,
        digital_artifacts=True,
        color_bleeding=True,
        screen_tear=True,
        use_gpu=True,
        batch_size=6,
        background_color="#000000",
        width=640,
        height=640,
        quality=90
    ):
        """生成视频故障艺术转场效果 - 赛博朋克数字故障"""
        
        start_time = time.time()
        print(f"🔥 开始生成故障艺术转场（赛博朋克风格）...")
        print(f"   故障样式: {glitch_style}")
        print(f"   故障强度: {glitch_intensity}")
        print(f"   损坏率: {corruption_rate}")
        print(f"   扫描线频率: {scanline_frequency}")
        print(f"   噪声量: {noise_amount}")
        print(f"   数字伪影: {'启用' if digital_artifacts else '禁用'}")
        print(f"   颜色溢出: {'启用' if color_bleeding else '禁用'}")
        print(f"   屏幕撕裂: {'启用' if screen_tear else '禁用'}")
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
        
        # 使用Playwright直接渲染故障效果
        from playwright.async_api import async_playwright
        
        playwright = await async_playwright().start()
        
        try:
            # 根据GPU设置选择渲染方式
            if use_gpu:
                # GPU硬件加速（故障效果需要GPU）
                browser = await playwright.chromium.launch(
                    headless=True,
                    args=[
                        '--enable-gpu',
                        '--use-gl=angle',
                        '--enable-webgl',
                        '--enable-webgl2',
                        '--enable-accelerated-2d-canvas',
                        '--disable-dev-shm-usage',
                        '--hide-scrollbars',
                        '--mute-audio',
                        '--no-sandbox',
                        '--disable-setuid-sandbox',
                        '--max_old_space_size=4096',
                    ]
                )
                print("✅ Playwright浏览器已启动（GPU硬件加速 + WebGL2）")
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
                glitch_style, glitch_intensity, corruption_rate,
                scanline_frequency, noise_amount, digital_artifacts, color_bleeding, 
                screen_tear, background_color, width, height
            )
            
            # 创建页面
            page = await browser.new_page(viewport={'width': width, 'height': height})
            
            try:
                # 加载HTML页面
                await page.set_content(html_content, wait_until='domcontentloaded', timeout=30000)
                
                # 等待故障系统初始化
                await page.wait_for_function("window.glitchController && window.glitchController.ready", timeout=15000)
                
                print("✅ 故障艺术系统初始化完成")
                
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
        print(f"✅ 故障艺术转场完成，输出: {video_tensor.shape}")
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
            
            # 更新故障动画
            await page.evaluate(f"""
                if (window.glitchController) {{
                    window.glitchController.updateFrame({progress}, '{frame1_base64}', '{frame2_base64}');
                }}
            """)
            
            # 等待故障系统更新（故障效果需要更多时间）
            await page.wait_for_timeout(40)
            
            # 强制触发重绘
            await page.evaluate("document.body.offsetHeight")
            
            # 等待渲染完成
            await page.wait_for_timeout(20)
            
            # 优化截图：使用JPEG格式
            screenshot_bytes = await page.screenshot(type='jpeg', quality=quality)
            
            # 转换为numpy数组
            from PIL import Image
            import io
            image = Image.open(io.BytesIO(screenshot_bytes)).convert('RGB')
            final_frame = np.array(image).astype(np.uint8)
            
            batch_frames.append(final_frame)
        
        return batch_frames
    
    def _generate_html_template(self, glitch_style, glitch_intensity, corruption_rate, scanline_frequency, noise_amount, digital_artifacts, color_bleeding, screen_tear, background_color, width, height):
        """生成HTML模板 - WebGL故障艺术系统效果（赛博朋克风格）"""
        
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
            font-family: 'Courier New', monospace;
        }}
        
        .glitch-container {{
            position: relative;
            width: 100%;
            height: 100%;
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
        
        .glitch-layer {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            mix-blend-mode: screen;
        }}
        
        .scanlines {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: repeating-linear-gradient(
                0deg,
                transparent,
                transparent 2px,
                rgba(0, 255, 0, 0.03) 2px,
                rgba(0, 255, 0, 0.03) 4px
            );
            pointer-events: none;
        }}
        
        .noise-overlay {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            opacity: 0.1;
            background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" viewBox="0 0 100 100"><defs><filter id="noise"><feTurbulence baseFrequency="0.9" numOctaves="4" stitchTiles="stitch"/></filter></defs><rect width="100%" height="100%" filter="url(%23noise)"/></svg>');
            pointer-events: none;
        }}
        
        .loading {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: #00ff00;
            font-family: 'Courier New', monospace;
            font-size: 14px;
            text-shadow: 0 0 10px #00ff00;
        }}
    </style>
</head>
<body>
    <div class="glitch-container" id="glitchContainer">
        <div class="video-layer" id="video1Layer"></div>
        <div class="video-layer" id="video2Layer"></div>
        <div class="glitch-layer" id="glitchLayer"></div>
        <div class="scanlines" id="scanlines"></div>
        <div class="noise-overlay" id="noiseOverlay"></div>
    </div>
    <div class="loading" id="loading">初始化故障艺术系统...</div>
    
    <script>
        class GlitchController {{
            constructor() {{
                this.ready = false;
                this.glitchStyle = '{glitch_style}';
                this.glitchIntensity = {glitch_intensity};
                this.corruptionRate = {corruption_rate};
                this.scanlineFrequency = {scanline_frequency};
                this.noiseAmount = {noise_amount};
                this.digitalArtifacts = {str(digital_artifacts).lower()};
                this.colorBleeding = {str(color_bleeding).lower()};
                this.screenTear = {str(screen_tear).lower()};
                this.width = {width};
                this.height = {height};
                
                this.video1Layer = null;
                this.video2Layer = null;
                this.glitchLayer = null;
                this.scanlines = null;
                this.noiseOverlay = null;
                
                this.glitchBlocks = [];
                
                this.init();
            }}
            
            async init() {{
                try {{
                    this.video1Layer = document.getElementById('video1Layer');
                    this.video2Layer = document.getElementById('video2Layer');
                    this.glitchLayer = document.getElementById('glitchLayer');
                    this.scanlines = document.getElementById('scanlines');
                    this.noiseOverlay = document.getElementById('noiseOverlay');
                    
                    await this.createGlitchElements();
                    this.ready = true;
                    document.getElementById('loading').style.display = 'none';
                    console.log('🔥 GlitchController初始化完成, 样式=' + this.glitchStyle);
                }} catch (error) {{
                    console.error('故障艺术系统初始化失败:', error);
                }}
            }}
            
            async createGlitchElements() {{
                // 创建故障块元素
                this.createGlitchBlocks();
                
                // 设置扫描线
                this.setupScanlines();
                
                // 设置噪声覆盖
                this.setupNoiseOverlay();
            }}
            
            createGlitchBlocks() {{
                // 创建故障块用于各种故障效果 - 增加数量以支持密集矩阵雨
                const blockCount = 50;  // 大幅增加故障块数量
                this.glitchBlocks = [];
                
                for (let i = 0; i < blockCount; i++) {{
                    const block = document.createElement('div');
                    block.style.position = 'absolute';
                    block.style.backgroundColor = this.getRandomGlitchColor();
                    block.style.mixBlendMode = 'difference';
                    block.style.opacity = '0';
                    block.style.pointerEvents = 'none';
                    
                    this.glitchLayer.appendChild(block);
                    this.glitchBlocks.push(block);
                }}
            }}
            
            setupScanlines() {{
                // 动态扫描线效果
                if (this.scanlines) {{
                    this.scanlines.style.animation = `scanlineMove ${{2 / this.scanlineFrequency}}s linear infinite`;
                    
                    // 添加扫描线动画CSS
                    const style = document.createElement('style');
                    style.textContent = `
                        @keyframes scanlineMove {{
                            0% {{ transform: translateY(-100%); }}
                            100% {{ transform: translateY(100vh); }}
                        }}
                    `;
                    document.head.appendChild(style);
                }}
            }}
            
            setupNoiseOverlay() {{
                // 动态噪声覆盖
                if (this.noiseOverlay) {{
                    this.noiseOverlay.style.opacity = this.noiseAmount;
                    this.noiseOverlay.style.animation = 'noiseFlicker 0.1s infinite';
                    
                    // 添加噪声闪烁动画CSS
                    const style = document.createElement('style');
                    style.textContent = `
                        @keyframes noiseFlicker {{
                            0%, 100% {{ opacity: ${{this.noiseAmount}}; }}
                            50% {{ opacity: ${{this.noiseAmount * 1.5}}; }}
                        }}
                    `;
                    document.head.appendChild(style);
                }}
            }}
            
            updateFrame(progress, texture1Base64, texture2Base64) {{
                if (!this.ready) return;
                
                try {{
                    // 更新视频图层
                    this.updateVideoLayers(progress, texture1Base64, texture2Base64);
                    
                    // 根据故障样式应用效果
                    switch(this.glitchStyle) {{
                        case 'signal_interference':
                            this.applySignalInterference(progress);
                            break;
                        case 'data_moshing':
                            this.applyDataMoshing(progress);
                            break;
                        case 'pixel_sorting':
                            this.applyPixelSorting(progress);
                            break;
                        case 'matrix_rain':
                            this.applyMatrixRain(progress);
                            break;
                    }}
                    
                    // 应用通用故障效果
                    this.applyCommonGlitchEffects(progress);
                }} catch (error) {{
                    console.error('故障更新失败:', error);
                }}
            }}
            
            updateVideoLayers(progress, texture1Base64, texture2Base64) {{
                // 更新视频图层背景
                this.video1Layer.style.backgroundImage = `url(${{texture1Base64}})`;
                this.video2Layer.style.backgroundImage = `url(${{texture2Base64}})`;
                
                // 矩阵雨模式：纯闪烁切换，无叠化过渡
                if (this.glitchStyle === 'matrix_rain') {{
                    // 基于进度的硬切换点
                    const switchPoint = 0.5; // 中点切换
                    const flickerZone = 0.3; // 闪烁区域范围
                    
                    if (progress < switchPoint - flickerZone/2) {{
                        // 前半段：显示第一个视频
                        this.video1Layer.style.opacity = '1';
                        this.video2Layer.style.opacity = '0';
                    }} else if (progress > switchPoint + flickerZone/2) {{
                        // 后半段：显示第二个视频
                        this.video1Layer.style.opacity = '0';
                        this.video2Layer.style.opacity = '1';
                    }} else {{
                        // 中间闪烁区域：快速切换
                        const flickerProgress = (progress - (switchPoint - flickerZone/2)) / flickerZone;
                        const flickerSpeed = 15; // 闪烁频率
                        const flicker = Math.sin(flickerProgress * Math.PI * flickerSpeed) > 0;
                        
                        if (flicker) {{
                            this.video1Layer.style.opacity = '1';
                            this.video2Layer.style.opacity = '0';
                        }} else {{
                            this.video1Layer.style.opacity = '0';
                            this.video2Layer.style.opacity = '1';
                        }}
                    }}
                }} else {{
                    // 其他模式：使用原来的叠化过渡
                    const glitchProgress = this.calculateGlitchProgress(progress);
                    this.video1Layer.style.opacity = 1 - glitchProgress;
                    this.video2Layer.style.opacity = glitchProgress;
                }}
            }}
            
            calculateGlitchProgress(progress) {{
                // 计算故障进度（非线性，有突变）
                const corruptionThreshold = this.corruptionRate;
                
                if (progress < corruptionThreshold) {{
                    return 0;
                }} else if (progress < corruptionThreshold + 0.2) {{
                    // 故障爆发期
                    const burstProgress = (progress - corruptionThreshold) / 0.2;
                    return Math.pow(burstProgress, 0.5); // 快速上升
                }} else {{
                    // 稳定期
                    const stableProgress = (progress - corruptionThreshold - 0.2) / (1 - corruptionThreshold - 0.2);
                    return 0.5 + stableProgress * 0.5;
                }}
            }}
            
            applySignalInterference(progress) {{
                // 信号干扰效果 - 增强版
                const interferenceStrength = Math.sin(progress * Math.PI * 10) * this.glitchIntensity;
                const pulseInterference = Math.sin(progress * Math.PI * 25) * interferenceStrength * 0.8;
                const totalInterference = interferenceStrength + pulseInterference;
                
                // 增强水平条纹干扰
                this.glitchBlocks.forEach((block, index) => {{
                    if (Math.random() < Math.abs(totalInterference) * 0.6) {{  // 提高触发概率
                        const stripeType = index % 4;
                        
                        switch(stripeType) {{
                            case 0: // 粗白色条纹
                                block.style.left = '0px';
                                block.style.top = (Math.random() * this.height) + 'px';
                                block.style.width = this.width + 'px';
                                block.style.height = (Math.random() * 12 + 8) + 'px';  // 更粗的条纹
                                block.style.backgroundColor = '#ffffff';
                                block.style.opacity = Math.random() * 0.9 + 0.3;  // 更高透明度
                                block.style.mixBlendMode = 'screen';
                                block.style.boxShadow = '0 0 5px #ffffff';  // 发光效果
                                break;
                                
                            case 1: // 黑色干扰条纹
                                block.style.left = '0px';
                                block.style.top = (Math.random() * this.height) + 'px';
                                block.style.width = this.width + 'px';
                                block.style.height = (Math.random() * 8 + 4) + 'px';
                                block.style.backgroundColor = '#000000';
                                block.style.opacity = Math.random() * 0.8 + 0.4;
                                block.style.mixBlendMode = 'multiply';
                                break;
                                
                            case 2: // 彩色信号条纹
                                block.style.left = '0px';
                                block.style.top = (Math.random() * this.height) + 'px';
                                block.style.width = this.width + 'px';
                                block.style.height = (Math.random() * 6 + 3) + 'px';
                                const signalColors = ['#ff0000', '#00ff00', '#0000ff', '#ffff00', '#ff00ff', '#00ffff'];
                                block.style.backgroundColor = signalColors[Math.floor(Math.random() * signalColors.length)];
                                block.style.opacity = Math.random() * 0.7 + 0.2;
                                block.style.mixBlendMode = 'overlay';
                                break;
                                
                            case 3: // 渐变干扰条纹
                                block.style.left = '0px';
                                block.style.top = (Math.random() * this.height) + 'px';
                                block.style.width = this.width + 'px';
                                block.style.height = (Math.random() * 10 + 5) + 'px';
                                block.style.background = `linear-gradient(90deg, 
                                    transparent 0%, 
                                    #ffffff 20%, 
                                    #000000 50%, 
                                    #ffffff 80%, 
                                    transparent 100%)`;
                                block.style.opacity = Math.random() * 0.8 + 0.3;
                                block.style.mixBlendMode = 'difference';
                                break;
                        }}
                        
                        setTimeout(() => {{
                            block.style.opacity = '0';
                            block.style.boxShadow = '';
                        }}, Math.random() * 100 + 30);  // 更短的显示时间，更频繁
                    }}
                }});
                
                // 增强整体图像偏移
                const offsetX = Math.sin(progress * Math.PI * 20) * totalInterference * 15;  // 更大的偏移
                const offsetY = Math.sin(progress * Math.PI * 15) * totalInterference * 5;   // 添加垂直偏移
                this.video1Layer.style.transform = `translate(${{offsetX}}px, ${{offsetY}}px)`;
                this.video2Layer.style.transform = `translate(${{-offsetX}}px, ${{-offsetY}}px)`;
                
                // 添加信号失真效果
                if (Math.random() < Math.abs(totalInterference) * 0.3) {{
                    this.video1Layer.style.filter = `contrast(${{1.2 + Math.abs(totalInterference) * 0.5}}) brightness(${{1.1 + totalInterference * 0.2}})`;
                    this.video2Layer.style.filter = `saturate(${{1.3 + Math.abs(totalInterference) * 0.4}})`;
                }}
            }}
            
            applyDataMoshing(progress) {{
                // 数据混合效果
                const moshIntensity = progress * this.glitchIntensity;
                
                // 创建马赛克效果
                this.video1Layer.style.filter = `contrast(${{1 + moshIntensity}}) saturate(${{1 + moshIntensity * 2}})`;
                this.video2Layer.style.filter = `hue-rotate(${{moshIntensity * 180}}deg) contrast(${{1.5}})`;
                
                // 随机块状故障
                this.glitchBlocks.forEach((block, index) => {{
                    if (Math.random() < moshIntensity * 0.1) {{
                        const size = Math.random() * 80 + 20;
                        block.style.left = (Math.random() * this.width) + 'px';
                        block.style.top = (Math.random() * this.height) + 'px';
                        block.style.width = size + 'px';
                        block.style.height = size + 'px';
                        block.style.backgroundColor = this.getRandomGlitchColor();
                        block.style.opacity = Math.random() * 0.6;
                        block.style.mixBlendMode = 'multiply';
                        
                        setTimeout(() => {{
                            block.style.opacity = '0';
                        }}, Math.random() * 300 + 100);
                    }}
                }});
            }}
            
            applyPixelSorting(progress) {{
                // 像素排序效果
                const sortIntensity = Math.sin(progress * Math.PI) * this.glitchIntensity;
                
                // 模拟像素排序的条纹效果
                this.glitchBlocks.forEach((block, index) => {{
                    if (index < 10 && Math.random() < sortIntensity * 0.3) {{
                        block.style.left = '0px';
                        block.style.top = (index * this.height / 10) + 'px';
                        block.style.width = this.width + 'px';
                        block.style.height = (this.height / 10) + 'px';
                        
                        // 创建渐变效果模拟像素排序
                        const gradient = `linear-gradient(90deg, 
                            transparent 0%, 
                            ${{this.getRandomGlitchColor()}} 30%, 
                            transparent 70%, 
                            ${{this.getRandomGlitchColor()}} 100%)`;
                        block.style.background = gradient;
                        block.style.opacity = sortIntensity * 0.4;
                        block.style.mixBlendMode = 'overlay';
                    }}
                }});
            }}
            
            applyMatrixRain(progress) {{
                // 矩阵雨效果 - 增强版
                const rainIntensity = progress * this.glitchIntensity;
                const pulseRain = Math.sin(progress * Math.PI * 8) * rainIntensity * 0.5;
                const totalRainIntensity = rainIntensity + pulseRain;
                
                // 创建密集细线矩阵雨滴
                this.glitchBlocks.forEach((block, index) => {{
                    if (Math.random() < totalRainIntensity * 0.9) {{  // 进一步提高触发概率，超密集
                        const rainType = index % 8;  // 增加更多细线类型
                        
                        switch(rainType) {{
                            case 0: // 超细绿色数字雨
                                block.style.left = (Math.random() * this.width) + 'px';
                                block.style.top = '-50px';
                                block.style.width = '1px';  // 超细线条
                                block.style.height = (Math.random() * 200 + 100) + 'px';
                                block.style.backgroundColor = '#00ff00';
                                block.style.opacity = Math.random() * 0.9 + 0.4;  // 恢复之前的透明度
                                block.style.mixBlendMode = 'screen';
                                block.style.boxShadow = '0 0 10px #00ff00, 0 0 20px #00ff00';  // 恢复强发光
                                block.style.animation = `matrixFall ${{Math.random() * 1.5 + 1}}s linear forwards`;
                                break;
                                
                            case 1: // 极细亮绿线
                                block.style.left = (Math.random() * this.width) + 'px';
                                block.style.top = '-30px';
                                block.style.width = '1px';  // 极细
                                block.style.height = (Math.random() * 150 + 80) + 'px';
                                block.style.backgroundColor = '#00ff00';
                                block.style.opacity = '1';
                                block.style.mixBlendMode = 'screen';
                                block.style.boxShadow = '0 0 15px #00ff00, 0 0 30px #00ff00, 0 0 45px #00ff00';
                                block.style.filter = 'brightness(2)';
                                block.style.animation = `matrixFall ${{Math.random() * 1.2 + 0.8}}s linear forwards`;
                                break;
                                
                            case 2: // 细线渐变雨
                                block.style.left = (Math.random() * this.width) + 'px';
                                block.style.top = '-40px';
                                block.style.width = '1px';  // 细线
                                block.style.height = (Math.random() * 180 + 90) + 'px';
                                block.style.background = `linear-gradient(180deg, 
                                    #00ff00 0%, 
                                    #00cc00 30%, 
                                    #008800 70%, 
                                    transparent 100%)`;
                                block.style.opacity = Math.random() * 0.8 + 0.3;
                                block.style.mixBlendMode = 'screen';
                                block.style.animation = `matrixFall ${{Math.random() * 1.8 + 1.2}}s linear forwards`;
                                break;
                                
                            case 3: // 短细线闪烁雨
                                block.style.left = (Math.random() * this.width) + 'px';
                                block.style.top = '-20px';
                                block.style.width = '1px';  // 细线
                                block.style.height = (Math.random() * 120 + 60) + 'px';
                                block.style.backgroundColor = '#00ff00';
                                block.style.opacity = Math.random() * 0.7 + 0.2;
                                block.style.mixBlendMode = 'screen';
                                block.style.animation = `matrixFall ${{Math.random() * 1.5 + 1}}s linear forwards, matrixFlicker 0.1s infinite`;
                                break;
                                
                            case 4: // 密集细雨滴
                                block.style.left = (Math.random() * this.width) + 'px';
                                block.style.top = '-60px';
                                block.style.width = '1px';  // 最细
                                block.style.height = (Math.random() * 300 + 150) + 'px';
                                block.style.backgroundColor = '#00ff00';
                                block.style.opacity = Math.random() * 0.9 + 0.5;
                                block.style.mixBlendMode = 'screen';
                                block.style.boxShadow = '0 0 20px #00ff00';
                                block.style.animation = `matrixFall ${{Math.random() * 2 + 1.5}}s linear forwards`;
                                break;
                                
                            case 5: // 超密集微细线
                                block.style.left = (Math.random() * this.width) + 'px';
                                block.style.top = '-35px';
                                block.style.width = '1px';  // 微细线
                                block.style.height = (Math.random() * 100 + 50) + 'px';
                                block.style.backgroundColor = '#00ff00';
                                block.style.opacity = Math.random() * 0.8 + 0.3;
                                block.style.mixBlendMode = 'screen';
                                block.style.animation = `matrixFall ${{Math.random() * 1.3 + 0.9}}s linear forwards`;
                                break;
                                
                            case 6: // 极短细线
                                block.style.left = (Math.random() * this.width) + 'px';
                                block.style.top = '-25px';
                                block.style.width = '1px';  // 极细
                                block.style.height = (Math.random() * 80 + 40) + 'px';
                                block.style.backgroundColor = '#00ff00';
                                block.style.opacity = Math.random() * 0.7 + 0.4;
                                block.style.mixBlendMode = 'screen';
                                block.style.animation = `matrixFall ${{Math.random() * 1.1 + 0.7}}s linear forwards`;
                                break;
                                
                            case 7: // 快速细线雨
                                block.style.left = (Math.random() * this.width) + 'px';
                                block.style.top = '-15px';
                                block.style.width = '1px';  // 细线
                                block.style.height = (Math.random() * 60 + 30) + 'px';
                                block.style.backgroundColor = '#00ff00';
                                block.style.opacity = Math.random() * 0.6 + 0.3;
                                block.style.mixBlendMode = 'screen';
                                block.style.animation = `matrixFall ${{Math.random() * 0.9 + 0.6}}s linear forwards, matrixFlicker 0.1s infinite`;
                                break;
                        }}
                        
                        setTimeout(() => {{
                            block.style.opacity = '0';
                            block.style.animation = '';
                            block.style.boxShadow = '';
                            block.style.filter = '';
                        }}, Math.random() * 2500 + 1500);  // 恢复之前的显示时间
                    }}
                }});
                
                // 添加增强矩阵动画
                if (!document.getElementById('matrixStyle')) {{
                    const style = document.createElement('style');
                    style.id = 'matrixStyle';
                    style.textContent = `
                        @keyframes matrixFall {{
                            from {{ 
                                transform: translateY(-100px); 
                                opacity: 0;
                            }}
                            10% {{
                                opacity: 1;
                            }}
                            90% {{
                                opacity: 1;
                            }}
                            to {{ 
                                transform: translateY(${{this.height + 200}}px); 
                                opacity: 0;
                            }}
                        }}
                        
                        @keyframes matrixFlicker {{
                            0%, 100% {{ opacity: 0.3; }}
                            50% {{ opacity: 1; }}
                        }}
                    `;
                    document.head.appendChild(style);
                }}
                
                // 添加矩阵背景效果
                if (Math.random() < totalRainIntensity * 0.2) {{
                    this.video1Layer.style.filter = `
                        contrast(${{1.1 + totalRainIntensity * 0.3}}) 
                        brightness(${{0.9 + totalRainIntensity * 0.2}})
                        hue-rotate(${{totalRainIntensity * 10}}deg)
                    `;
                    this.video2Layer.style.filter = `
                        contrast(${{1.2 + totalRainIntensity * 0.2}})
                        saturate(${{0.8 + totalRainIntensity * 0.4}})
                    `;
                }}
                
                // 添加绿色闪光效果
                if (Math.random() < totalRainIntensity * 0.1) {{
                    document.body.style.backgroundColor = '#003300';
                    setTimeout(() => {{
                        document.body.style.backgroundColor = '{background_color}';
                    }}, 50);
                }}
            }}
            
            applyCommonGlitchEffects(progress) {{
                // 应用通用故障效果
                
                // 屏幕撕裂
                if (this.screenTear && Math.random() < progress * 0.1) {{
                    const tearHeight = Math.random() * 50 + 10;
                    const tearY = Math.random() * (this.height - tearHeight);
                    
                    this.video1Layer.style.clipPath = `polygon(0 0, 100% 0, 100% ${{tearY}}px, 0 ${{tearY}}px)`;
                    this.video2Layer.style.clipPath = `polygon(0 ${{tearY + tearHeight}}px, 100% ${{tearY + tearHeight}}px, 100% 100%, 0 100%)`;
                    
                    setTimeout(() => {{
                        this.video1Layer.style.clipPath = '';
                        this.video2Layer.style.clipPath = '';
                    }}, 100);
                }}
                
                // 颜色溢出
                if (this.colorBleeding) {{
                    const bleedIntensity = Math.sin(progress * Math.PI * 5) * this.glitchIntensity * 0.1;
                    this.video1Layer.style.filter += ` saturate(${{1 + bleedIntensity}})`;
                    this.video2Layer.style.filter += ` hue-rotate(${{bleedIntensity * 30}}deg)`;
                }}
            }}
            
            getRandomGlitchColor() {{
                // 获取随机故障颜色
                const colors = [
                    '#ff0000', '#00ff00', '#0000ff',
                    '#ff00ff', '#ffff00', '#00ffff',
                    '#ffffff', '#000000'
                ];
                return colors[Math.floor(Math.random() * colors.length)];
            }}
        }}
        
        // 初始化控制器
        window.glitchController = new GlitchController();
        console.log('✅ GlitchController initialized');
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
    "VideoGlitchArtTransitionNode": VideoGlitchArtTransitionNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoGlitchArtTransitionNode": "Video Glitch Art Transition",
}
