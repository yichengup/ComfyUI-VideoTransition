"""
视频故障艺术转场节点 - 增强版（超强视觉效果）
数字故障、像素错位、RGB分离、信号干扰等效果 - 加强版
"""

import torch
import json
import cv2
import numpy as np
import math
import time
from comfy.comfy_types.node_typing import ComfyNodeABC, InputTypeDict, IO


class VideoGlitchArtEnhancedNode(ComfyNodeABC):
    """视频故障艺术转场 - 增强版（超强视觉冲击）"""
    
    CATEGORY = "VideoTransition"
    
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "video1": (IO.IMAGE,),  # 第一个视频（故障源）
                "video2": (IO.IMAGE,),  # 第二个视频（目标）
                "glitch_style": ([
                    "compression_nightmare",   # 压缩噩梦
                    "digital_meltdown",        # 数字崩溃
                    "quantum_decay",           # 量子衰变
                    "neural_collapse",         # 神经网络崩溃
                ],),
                "total_frames": (IO.INT, {"default": 36, "min": 4, "max": 120}),
                "fps": (IO.INT, {"default": 24, "min": 15, "max": 60}),
            },
            "optional": {
                "chaos_level": (IO.FLOAT, {"default": 2.0, "min": 0.5, "max": 5.0}),
                "corruption_rate": (IO.FLOAT, {"default": 0.6, "min": 0.2, "max": 1.0}),
                "visual_intensity": (IO.FLOAT, {"default": 3.0, "min": 1.0, "max": 5.0}),
                "color_madness": (IO.FLOAT, {"default": 2.5, "min": 1.0, "max": 4.0}),
                "screen_tear": (IO.BOOLEAN, {"default": True}),
                "flash_effects": (IO.BOOLEAN, {"default": True}),
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
    FUNCTION = "generate_enhanced_glitch_transition"
    
    async def generate_enhanced_glitch_transition(
        self, 
        video1, 
        video2,
        glitch_style,
        total_frames, 
        fps,
        chaos_level=2.0,
        corruption_rate=0.6,
        visual_intensity=3.0,
        color_madness=2.5,
        screen_tear=True,
        flash_effects=True,
        use_gpu=True,
        batch_size=6,
        background_color="#000000",
        width=640,
        height=640,
        quality=90
    ):
        """生成增强版故障艺术转场效果"""
        
        start_time = time.time()
        print(f"Starting enhanced glitch transition: {glitch_style}, {total_frames} frames")
        
        # 提取视频帧
        frames1 = self._extract_video_frames(video1)
        frames2 = self._extract_video_frames(video2)
        print(f"Video frames: {len(frames1)} -> {len(frames2)}, generating {total_frames} transition frames")
        
        # 预计算帧数据
        frame1_base64_list = []
        frame2_base64_list = []
        
        for i in range(total_frames):
            progress = i / (total_frames - 1) if total_frames > 1 else 0
            
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
            
            frame1_base64_list.append(self._numpy_to_base64(frame1_data))
            frame2_base64_list.append(self._numpy_to_base64(frame2_data))
        
        # 使用Playwright渲染
        from playwright.async_api import async_playwright
        
        playwright = await async_playwright().start()
        
        try:
            if use_gpu:
                print("Playwright browser starting with GPU acceleration")
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
            else:
                print("Playwright browser starting with SwiftShader (CPU rendering)")
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
            
            # 生成增强版HTML模板
            html_content = self._generate_enhanced_html_template(
                glitch_style, chaos_level, corruption_rate, visual_intensity,
                color_madness, screen_tear, flash_effects, background_color, width, height
            )
            
            page = await browser.new_page(viewport={'width': width, 'height': height})
            
            try:
                await page.set_content(html_content, wait_until='domcontentloaded', timeout=30000)
                await page.wait_for_function("window.enhancedGlitchController && window.enhancedGlitchController.ready", timeout=15000)
                
                output_frames = []
                render_start = time.time()
                
                for batch_start in range(0, total_frames, batch_size):
                    batch_end = min(batch_start + batch_size, total_frames)
                    batch_indices = list(range(batch_start, batch_end))
                    
                    batch_frames = await self._process_enhanced_batch(
                        page, batch_indices, frame1_base64_list, frame2_base64_list, 
                        total_frames, quality
                    )
                    
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
            
            finally:
                await page.close()
            
            await browser.close()
        finally:
            await playwright.stop()
        
        video_tensor = self._frames_to_tensor(output_frames)
        
        total_time = time.time() - start_time
        print(f"Enhanced glitch transition completed: {video_tensor.shape} in {total_time:.2f}s")
        
        return (video_tensor,)
    
    async def _process_enhanced_batch(self, page, batch_indices, frame1_base64_list, frame2_base64_list, total_frames, quality):
        """处理增强版批次"""
        batch_frames = []
        
        for i in batch_indices:
            progress = i / (total_frames - 1) if total_frames > 1 else 0
            
            frame1_base64 = frame1_base64_list[i]
            frame2_base64 = frame2_base64_list[i]
            
            await page.evaluate(f"""
                if (window.enhancedGlitchController) {{
                    window.enhancedGlitchController.updateFrame({progress}, '{frame1_base64}', '{frame2_base64}');
                }}
            """)
            
            await page.wait_for_timeout(40)
            await page.evaluate("document.body.offsetHeight")
            await page.wait_for_timeout(20)
            
            screenshot_bytes = await page.screenshot(type='jpeg', quality=quality)
            
            from PIL import Image
            import io
            image = Image.open(io.BytesIO(screenshot_bytes)).convert('RGB')
            final_frame = np.array(image).astype(np.uint8)
            
            batch_frames.append(final_frame)
        
        return batch_frames
    
    def _generate_enhanced_html_template(self, glitch_style, chaos_level, corruption_rate, visual_intensity, color_madness, screen_tear, flash_effects, background_color, width, height):
        """生成增强版HTML模板"""
        
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
        
        .enhanced-glitch-container {{
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
        
        .chaos-layer {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
        }}
        
        .loading {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: #ff0080;
            font-family: 'Courier New', monospace;
            font-size: 14px;
            text-shadow: 0 0 20px #ff0080;
            animation: loadingPulse 0.5s infinite alternate;
        }}
        
        @keyframes loadingPulse {{
            from {{ opacity: 0.5; }}
            to {{ opacity: 1.0; }}
        }}
    </style>
</head>
<body>
    <div class="enhanced-glitch-container" id="enhancedGlitchContainer">
        <div class="video-layer" id="video1Layer"></div>
        <div class="video-layer" id="video2Layer"></div>
        <div class="chaos-layer" id="chaosLayer"></div>
    </div>
    <div class="loading" id="loading">Initializing enhanced glitch system...</div>
    
    <script>
        class EnhancedGlitchController {{
            constructor() {{
                this.ready = false;
                this.glitchStyle = '{glitch_style}';
                this.chaosLevel = {chaos_level};
                this.corruptionRate = {corruption_rate};
                this.visualIntensity = {visual_intensity};
                this.colorMadness = {color_madness};
                this.screenTear = {str(screen_tear).lower()};
                this.flashEffects = {str(flash_effects).lower()};
                this.width = {width};
                this.height = {height};
                
                this.video1Layer = null;
                this.video2Layer = null;
                this.chaosLayer = null;
                
                this.chaosElements = [];
                
                this.init();
            }}
            
            async init() {{
                try {{
                    this.video1Layer = document.getElementById('video1Layer');
                    this.video2Layer = document.getElementById('video2Layer');
                    this.chaosLayer = document.getElementById('chaosLayer');
                    
                    await this.createChaosElements();
                    this.ready = true;
                    document.getElementById('loading').style.display = 'none';
                    // console.log('🔥💥 EnhancedGlitchController initialized');
                }} catch (error) {{
                    console.error('Enhanced glitch system init failed:', error);
                }}
            }}
            
            async createChaosElements() {{
                // 创建50个混乱元素
                for (let i = 0; i < 50; i++) {{
                    const element = document.createElement('div');
                    element.style.position = 'absolute';
                    element.style.pointerEvents = 'none';
                    this.chaosLayer.appendChild(element);
                    this.chaosElements.push(element);
                }}
            }}
            
            updateFrame(progress, texture1Base64, texture2Base64) {{
                if (!this.ready) return;
                
                try {{
                    this.updateVideoLayers(progress, texture1Base64, texture2Base64);
                    
                    switch(this.glitchStyle) {{
                        case 'compression_nightmare':
                            this.applyCompressionNightmare(progress);
                            break;
                        case 'digital_meltdown':
                            this.applyDigitalMeltdown(progress);
                            break;
                        case 'quantum_decay':
                            this.applyQuantumDecay(progress);
                            break;
                        case 'neural_collapse':
                            this.applyNeuralCollapse(progress);
                            break;
                    }}
                    
                    this.applyGlobalChaos(progress);
                }} catch (error) {{
                    console.error('Enhanced glitch update failed:', error);
                }}
            }}
            
            updateVideoLayers(progress, texture1Base64, texture2Base64) {{
                this.video1Layer.style.backgroundImage = `url(${{texture1Base64}})`;
                this.video2Layer.style.backgroundImage = `url(${{texture2Base64}})`;
                
                const chaosProgress = this.calculateChaosProgress(progress);
                this.video1Layer.style.opacity = 1 - chaosProgress;
                this.video2Layer.style.opacity = chaosProgress;
            }}
            
            calculateChaosProgress(progress) {{
                const threshold = this.corruptionRate;
                if (progress < threshold) {{
                    return Math.pow(progress / threshold, 0.3) * 0.2;
                }} else {{
                    const chaosProgress = (progress - threshold) / (1 - threshold);
                    return 0.2 + Math.pow(chaosProgress, 0.5) * 0.8;
                }}
            }}
            
            applyQuantumDecay(progress) {{
                // 量子衰变效果 - 模拟量子态坍缩和粒子衰变
                const decayIntensity = progress * this.visualIntensity;
                const quantumFluctuation = Math.sin(progress * Math.PI * 50) * decayIntensity * 0.3;
                const totalDecay = decayIntensity + quantumFluctuation;
                
                // 量子态坍缩滤镜
                this.video1Layer.style.filter = `
                    blur(${{totalDecay * 3}}px)
                    contrast(${{3 + totalDecay}})
                    brightness(${{0.5 + Math.abs(quantumFluctuation) * 0.8}})
                    saturate(${{0.2 + totalDecay * 0.5}})
                    hue-rotate(${{quantumFluctuation * 360}}deg)
                    invert(${{Math.abs(quantumFluctuation) * 0.3}})
                `;
                
                this.video2Layer.style.filter = `
                    contrast(${{2 + totalDecay * 0.8}})
                    brightness(${{1.2 + quantumFluctuation * 0.4}})
                    saturate(${{1.5 + totalDecay}})
                    sepia(${{totalDecay * 0.3}})
                `;
                
                // 量子粒子衰变效果
                this.chaosElements.forEach((element, index) => {{
                    if (Math.random() < totalDecay * 0.8) {{
                        const particleType = index % 6;
                        
                        switch(particleType) {{
                            case 0: // 电子云衰变
                                const electronSize = Math.random() * 30 + 10;
                                element.style.left = (Math.random() * this.width) + 'px';
                                element.style.top = (Math.random() * this.height) + 'px';
                                element.style.width = electronSize + 'px';
                                element.style.height = electronSize + 'px';
                                element.style.borderRadius = '50%';
                                element.style.background = `
                                    radial-gradient(circle,
                                        #00ffff 0%,
                                        #0080ff 30%,
                                        transparent 70%)
                                `;
                                element.style.opacity = Math.random() * 0.9 + 0.3;
                                element.style.mixBlendMode = 'screen';
                                element.style.boxShadow = '0 0 20px #00ffff, 0 0 40px #0080ff';
                                element.style.animation = 'quantumSpin 0.5s linear infinite';
                                break;
                                
                            case 1: // 质子衰变轨迹
                                element.style.left = (Math.random() * this.width) + 'px';
                                element.style.top = (Math.random() * this.height) + 'px';
                                element.style.width = (Math.random() * 200 + 100) + 'px';
                                element.style.height = '3px';
                                element.style.background = `
                                    linear-gradient(90deg,
                                        #ff4000 0%,
                                        #ff8000 50%,
                                        transparent 100%)
                                `;
                                element.style.opacity = Math.random() * 0.8 + 0.4;
                                element.style.mixBlendMode = 'screen';
                                element.style.transform = `rotate(${{Math.random() * 360}}deg)`;
                                element.style.boxShadow = '0 0 10px #ff4000';
                                break;
                                
                            case 2: // 中子星爆发
                                const neutronSize = Math.random() * 80 + 40;
                                element.style.left = (Math.random() * this.width) + 'px';
                                element.style.top = (Math.random() * this.height) + 'px';
                                element.style.width = neutronSize + 'px';
                                element.style.height = neutronSize + 'px';
                                element.style.background = `
                                    conic-gradient(
                                        from ${{Math.random() * 360}}deg,
                                        #ffffff 0deg,
                                        #ff00ff 60deg,
                                        #00ff00 120deg,
                                        #ffff00 180deg,
                                        #0000ff 240deg,
                                        #ff0000 300deg,
                                        #ffffff 360deg
                                    )
                                `;
                                element.style.opacity = Math.random() * 0.7 + 0.2;
                                element.style.mixBlendMode = 'overlay';
                                element.style.borderRadius = '50%';
                                element.style.filter = `blur(${{Math.random() * 8}}px)`;
                                element.style.animation = 'quantumPulse 0.3s ease-in-out infinite alternate';
                                break;
                                
                            case 3: // 量子隧道效应
                                element.style.left = '0px';
                                element.style.top = (Math.random() * this.height) + 'px';
                                element.style.width = this.width + 'px';
                                element.style.height = (Math.random() * 5 + 2) + 'px';
                                element.style.background = `
                                    repeating-linear-gradient(90deg,
                                        #00ffff 0px,
                                        transparent 3px,
                                        #ff00ff 6px,
                                        transparent 9px)
                                `;
                                element.style.opacity = Math.random() * 0.9 + 0.3;
                                element.style.mixBlendMode = 'difference';
                                element.style.animation = 'quantumTunnel 0.8s linear infinite';
                                break;
                                
                            case 4: // 波粒二象性
                                const waveSize = Math.random() * 150 + 50;
                                element.style.left = (Math.random() * this.width) + 'px';
                                element.style.top = (Math.random() * this.height) + 'px';
                                element.style.width = waveSize + 'px';
                                element.style.height = waveSize + 'px';
                                element.style.background = `
                                    repeating-radial-gradient(
                                        circle at 50% 50%,
                                        #ffff00 0px,
                                        transparent 8px,
                                        #ff0080 16px,
                                        transparent 24px
                                    )
                                `;
                                element.style.opacity = Math.random() * 0.6 + 0.2;
                                element.style.mixBlendMode = 'color-dodge';
                                element.style.borderRadius = '50%';
                                element.style.animation = 'quantumWave 1s ease-in-out infinite';
                                break;
                                
                            case 5: // 量子纠缠
                                const entangleSize = Math.random() * 60 + 20;
                                element.style.left = (Math.random() * this.width) + 'px';
                                element.style.top = (Math.random() * this.height) + 'px';
                                element.style.width = entangleSize + 'px';
                                element.style.height = entangleSize + 'px';
                                element.style.background = `
                                    linear-gradient(45deg,
                                        #ff0000 0%,
                                        #00ff00 25%,
                                        #0000ff 50%,
                                        #ff00ff 75%,
                                        #ffff00 100%)
                                `;
                                element.style.opacity = Math.random() * 0.8 + 0.4;
                                element.style.mixBlendMode = 'hard-light';
                                element.style.transform = `rotate(${{Math.random() * 360}}deg) scale(${{0.5 + Math.random() * 1}})`;
                                element.style.filter = `blur(${{Math.random() * 6}}px)`;
                                element.style.boxShadow = `0 0 30px ${{this.getRandomNightmareColor()}}`;
                                break;
                        }}
                        
                        setTimeout(() => {{
                            element.style.opacity = '0';
                            element.style.animation = '';
                            element.style.transform = '';
                            element.style.filter = '';
                            element.style.boxShadow = '';
                            element.style.borderRadius = '';
                        }}, Math.random() * 800 + 400);
                    }}
                }});
                
                // 量子真空涨落
                if (Math.random() < totalDecay * 0.3) {{
                    document.body.style.background = `
                        radial-gradient(circle at ${{Math.random() * 100}}% ${{Math.random() * 100}}%,
                            ${{this.getRandomNightmareColor()}} 0%,
                            transparent 20%)
                    `;
                    setTimeout(() => {{
                        document.body.style.background = '{background_color}';
                    }}, 100);
                }}
            }}
            
            applyNeuralCollapse(progress) {{
                // 神经网络崩溃效果 - 模拟AI神经网络失效和权重崩塌
                const collapseIntensity = progress * this.visualIntensity;
                const neuralSpike = Math.sin(progress * Math.PI * 40) * collapseIntensity * 0.4;
                const totalCollapse = collapseIntensity + neuralSpike;
                
                // 神经网络权重失效滤镜
                this.video1Layer.style.filter = `
                    contrast(${{1 + totalCollapse * 3}})
                    brightness(${{0.3 + Math.abs(neuralSpike) * 1.2}})
                    saturate(${{0.1 + totalCollapse * 0.8}})
                    blur(${{totalCollapse * 5}}px)
                    hue-rotate(${{neuralSpike * 720}}deg)
                    grayscale(${{totalCollapse * 0.7}})
                `;
                
                this.video2Layer.style.filter = `
                    contrast(${{2 + totalCollapse * 2}})
                    brightness(${{1.5 + neuralSpike * 0.6}})
                    saturate(${{2 + totalCollapse}})
                    sepia(${{totalCollapse * 0.5}})
                `;
                
                // 神经元连接断裂效果
                this.chaosElements.forEach((element, index) => {{
                    if (Math.random() < totalCollapse * 0.9) {{
                        const neuronType = index % 7;
                        
                        switch(neuronType) {{
                            case 0: // 神经元节点崩溃
                                const neuronSize = Math.random() * 40 + 15;
                                element.style.left = (Math.random() * this.width) + 'px';
                                element.style.top = (Math.random() * this.height) + 'px';
                                element.style.width = neuronSize + 'px';
                                element.style.height = neuronSize + 'px';
                                element.style.borderRadius = '50%';
                                element.style.background = `
                                    radial-gradient(circle,
                                        #ff0040 0%,
                                        #ff4080 40%,
                                        transparent 80%)
                                `;
                                element.style.opacity = Math.random() * 0.9 + 0.4;
                                element.style.mixBlendMode = 'screen';
                                element.style.boxShadow = '0 0 25px #ff0040, 0 0 50px #ff4080';
                                element.style.animation = 'neuronPulse 0.4s ease-in-out infinite alternate';
                                break;
                                
                            case 1: // 突触连接断裂
                                const synapseLength = Math.random() * 300 + 100;
                                element.style.left = (Math.random() * this.width) + 'px';
                                element.style.top = (Math.random() * this.height) + 'px';
                                element.style.width = synapseLength + 'px';
                                element.style.height = '4px';
                                element.style.background = `
                                    linear-gradient(90deg,
                                        #00ff80 0%,
                                        #ff0080 30%,
                                        transparent 50%,
                                        #8000ff 70%,
                                        transparent 100%)
                                `;
                                element.style.opacity = Math.random() * 0.8 + 0.3;
                                element.style.mixBlendMode = 'overlay';
                                element.style.transform = `rotate(${{Math.random() * 360}}deg)`;
                                element.style.boxShadow = '0 0 15px #00ff80';
                                element.style.animation = 'synapseFlicker 0.2s infinite';
                                break;
                                
                            case 2: // 权重矩阵错乱
                                const matrixSize = Math.random() * 120 + 60;
                                element.style.left = (Math.random() * this.width) + 'px';
                                element.style.top = (Math.random() * this.height) + 'px';
                                element.style.width = matrixSize + 'px';
                                element.style.height = matrixSize + 'px';
                                element.style.background = `
                                    repeating-conic-gradient(
                                        from ${{Math.random() * 360}}deg at 50% 50%,
                                        #ff8000 0deg 30deg,
                                        #00ff80 30deg 60deg,
                                        #8000ff 60deg 90deg,
                                        #ff0080 90deg 120deg
                                    )
                                `;
                                element.style.opacity = Math.random() * 0.7 + 0.2;
                                element.style.mixBlendMode = 'hard-light';
                                element.style.filter = `blur(${{Math.random() * 6}}px)`;
                                element.style.animation = 'matrixRotate 1.5s linear infinite';
                                break;
                                
                            case 3: // 激活函数失效
                                element.style.left = '0px';
                                element.style.top = (Math.random() * this.height) + 'px';
                                element.style.width = this.width + 'px';
                                element.style.height = (Math.random() * 12 + 6) + 'px';
                                element.style.background = `
                                    repeating-linear-gradient(90deg,
                                        #ff4000 0px,
                                        #00ff40 8px,
                                        #4000ff 16px,
                                        #ff0040 24px)
                                `;
                                element.style.opacity = Math.random() * 0.8 + 0.4;
                                element.style.mixBlendMode = 'difference';
                                element.style.animation = 'activationGlitch 0.3s linear infinite';
                                break;
                                
                            case 4: // 反向传播错误
                                const backpropSize = Math.random() * 200 + 80;
                                element.style.left = (Math.random() * this.width) + 'px';
                                element.style.top = (Math.random() * this.height) + 'px';
                                element.style.width = backpropSize + 'px';
                                element.style.height = backpropSize + 'px';
                                element.style.background = `
                                    conic-gradient(
                                        from 0deg,
                                        #ff0000 0deg,
                                        #ff8000 45deg,
                                        #ffff00 90deg,
                                        #80ff00 135deg,
                                        #00ff00 180deg,
                                        #00ff80 225deg,
                                        #00ffff 270deg,
                                        #8000ff 315deg,
                                        #ff0000 360deg
                                    )
                                `;
                                element.style.opacity = Math.random() * 0.6 + 0.3;
                                element.style.mixBlendMode = 'color-burn';
                                element.style.borderRadius = '50%';
                                element.style.filter = `blur(${{Math.random() * 10}}px)`;
                                element.style.animation = 'backpropSpin 2s ease-in-out infinite reverse';
                                break;
                                
                            case 5: // 梯度爆炸
                                element.style.left = (Math.random() * this.width) + 'px';
                                element.style.top = (Math.random() * this.height) + 'px';
                                element.style.width = (Math.random() * 8 + 4) + 'px';
                                element.style.height = (Math.random() * 8 + 4) + 'px';
                                element.style.backgroundColor = '#ffffff';
                                element.style.opacity = '1';
                                element.style.mixBlendMode = 'screen';
                                element.style.borderRadius = '50%';
                                element.style.boxShadow = `
                                    0 0 20px #ffffff,
                                    0 0 40px #ff0080,
                                    0 0 60px #00ff80,
                                    0 0 80px #8000ff
                                `;
                                element.style.animation = 'gradientExplode 0.1s ease-out infinite';
                                break;
                                
                            case 6: // 过拟合崩溃
                                const overfitSize = Math.random() * 150 + 50;
                                element.style.left = (Math.random() * this.width) + 'px';
                                element.style.top = (Math.random() * this.height) + 'px';
                                element.style.width = overfitSize + 'px';
                                element.style.height = overfitSize + 'px';
                                element.style.background = `
                                    repeating-radial-gradient(
                                        circle at 50% 50%,
                                        #ff0080 0px,
                                        transparent 6px,
                                        #00ff80 12px,
                                        transparent 18px,
                                        #8000ff 24px,
                                        transparent 30px
                                    )
                                `;
                                element.style.opacity = Math.random() * 0.7 + 0.2;
                                element.style.mixBlendMode = 'luminosity';
                                element.style.borderRadius = '50%';
                                element.style.animation = 'overfitCollapse 1s ease-in-out infinite alternate';
                                break;
                        }}
                        
                        setTimeout(() => {{
                            element.style.opacity = '0';
                            element.style.animation = '';
                            element.style.transform = '';
                            element.style.filter = '';
                            element.style.boxShadow = '';
                            element.style.borderRadius = '';
                        }}, Math.random() * 600 + 300);
                    }}
                }});
                
                // 神经网络系统性崩溃
                if (Math.random() < totalCollapse * 0.4) {{
                    document.body.style.background = `
                        linear-gradient(${{Math.random() * 360}}deg,
                            #ff0040 0%,
                            #00ff80 25%,
                            #8000ff 50%,
                            #ff8000 75%,
                            #ff0040 100%)
                    `;
                    document.body.style.filter = `blur(${{totalCollapse * 2}}px)`;
                    setTimeout(() => {{
                        document.body.style.background = '{background_color}';
                        document.body.style.filter = '';
                    }}, 150);
                }}
                
                // AI意识闪烁
                if (this.flashEffects && Math.random() < neuralSpike * 0.3) {{
                    const consciousnessColors = ['#ff0040', '#00ff80', '#8000ff', '#ff8000', '#40ff00'];
                    document.body.style.backgroundColor = consciousnessColors[Math.floor(Math.random() * consciousnessColors.length)];
                    setTimeout(() => {{
                        document.body.style.backgroundColor = '{background_color}';
                    }}, 80);
                }}
            }}
            
            applyCompressionNightmare(progress) {{
                // 压缩噩梦效果
                const nightmareIntensity = progress * this.visualIntensity;
                const compressionPulse = Math.sin(progress * Math.PI * 12) * nightmareIntensity;
                
                // 极端压缩失真
                this.video1Layer.style.filter = `
                    blur(${{nightmareIntensity * 8}}px) 
                    contrast(${{2 + nightmareIntensity * 2}}) 
                    saturate(${{0.3 + nightmareIntensity}})
                    brightness(${{1.2 + compressionPulse * 0.5}})
                    hue-rotate(${{compressionPulse * 180}}deg)
                `;
                
                this.video2Layer.style.filter = `
                    contrast(${{3 + nightmareIntensity}})
                    saturate(${{2 + nightmareIntensity * 2}})
                    brightness(${{0.8 + nightmareIntensity * 0.5}})
                `;
                
                // 创建压缩噩梦块
                this.chaosElements.forEach((element, index) => {{
                    if (Math.random() < nightmareIntensity * 0.5) {{
                        const nightmareType = index % 8;
                        
                        switch(nightmareType) {{
                            case 0: // 4x4 超小块
                                const size4 = 4;
                                element.style.left = (Math.floor(Math.random() * this.width / size4) * size4) + 'px';
                                element.style.top = (Math.floor(Math.random() * this.height / size4) * size4) + 'px';
                                element.style.width = size4 + 'px';
                                element.style.height = size4 + 'px';
                                element.style.backgroundColor = this.getRandomNightmareColor();
                                element.style.opacity = nightmareIntensity;
                                element.style.mixBlendMode = 'multiply';
                                break;
                                
                            case 1: // 32x32 大块
                                const size32 = 32;
                                element.style.left = (Math.floor(Math.random() * this.width / size32) * size32) + 'px';
                                element.style.top = (Math.floor(Math.random() * this.height / size32) * size32) + 'px';
                                element.style.width = size32 + 'px';
                                element.style.height = size32 + 'px';
                                element.style.background = `
                                    repeating-conic-gradient(
                                        from 0deg at 50% 50%,
                                        ${{this.getRandomNightmareColor()}} 0deg 45deg,
                                        ${{this.getRandomNightmareColor()}} 45deg 90deg,
                                        ${{this.getRandomNightmareColor()}} 90deg 135deg,
                                        ${{this.getRandomNightmareColor()}} 135deg 180deg
                                    )
                                `;
                                element.style.opacity = nightmareIntensity * 0.7;
                                element.style.mixBlendMode = 'overlay';
                                break;
                                
                            case 2: // 色度噩梦
                                element.style.left = (Math.random() * this.width) + 'px';
                                element.style.top = (Math.random() * this.height) + 'px';
                                element.style.width = (Math.random() * 80 + 40) + 'px';
                                element.style.height = (Math.random() * 80 + 40) + 'px';
                                element.style.background = `
                                    radial-gradient(ellipse, 
                                        ${{this.getRandomNightmareColor()}} 0%, 
                                        ${{this.getRandomNightmareColor()}} 30%,
                                        transparent 70%)
                                `;
                                element.style.opacity = nightmareIntensity * 0.8;
                                element.style.mixBlendMode = 'color-burn';
                                element.style.filter = `blur(${{Math.random() * 10}}px)`;
                                break;
                                
                            case 3: // 量化噩梦条纹
                                element.style.left = '0px';
                                element.style.top = (Math.random() * this.height) + 'px';
                                element.style.width = this.width + 'px';
                                element.style.height = (Math.random() * 8 + 3) + 'px';
                                element.style.background = `
                                    repeating-linear-gradient(90deg,
                                        ${{this.getRandomNightmareColor()}} 0px,
                                        ${{this.getRandomNightmareColor()}} 2px,
                                        ${{this.getRandomNightmareColor()}} 4px,
                                        ${{this.getRandomNightmareColor()}} 6px)
                                `;
                                element.style.opacity = nightmareIntensity * 0.6;
                                element.style.mixBlendMode = 'hard-light';
                                break;
                                
                            case 4: // 运动补偿噩梦
                                const motionSize = Math.random() * 100 + 50;
                                element.style.left = (Math.random() * this.width) + 'px';
                                element.style.top = (Math.random() * this.height) + 'px';
                                element.style.width = motionSize + 'px';
                                element.style.height = motionSize + 'px';
                                element.style.backgroundColor = this.getRandomNightmareColor();
                                element.style.opacity = nightmareIntensity;
                                element.style.mixBlendMode = 'difference';
                                element.style.filter = `blur(${{nightmareIntensity * 8}}px)`;
                                element.style.transform = `rotate(${{Math.random() * 360}}deg)`;
                                break;
                                
                            case 5: // DCT系数噩梦
                                element.style.left = (Math.random() * this.width) + 'px';
                                element.style.top = (Math.random() * this.height) + 'px';
                                element.style.width = '8px';
                                element.style.height = '8px';
                                element.style.backgroundColor = this.getRandomNightmareColor();
                                element.style.opacity = '1';
                                element.style.mixBlendMode = 'exclusion';
                                element.style.boxShadow = `
                                    0 0 20px ${{this.getRandomNightmareColor()}},
                                    0 0 40px ${{this.getRandomNightmareColor()}}
                                `;
                                break;
                                
                            case 6: // 帧间预测噩梦
                                element.style.left = '0px';
                                element.style.top = '0px';
                                element.style.width = this.width + 'px';
                                element.style.height = this.height + 'px';
                                element.style.background = `
                                    conic-gradient(
                                        from ${{Math.random() * 360}}deg,
                                        transparent 0deg,
                                        ${{this.getRandomNightmareColor()}} 90deg,
                                        transparent 180deg,
                                        ${{this.getRandomNightmareColor()}} 270deg,
                                        transparent 360deg
                                    )
                                `;
                                element.style.opacity = nightmareIntensity * 0.3;
                                element.style.mixBlendMode = 'color-dodge';
                                break;
                                
                            case 7: // 熵编码噩梦
                                element.style.left = (Math.random() * this.width) + 'px';
                                element.style.top = (Math.random() * this.height) + 'px';
                                element.style.width = (Math.random() * 200 + 50) + 'px';
                                element.style.height = (Math.random() * 200 + 50) + 'px';
                                element.style.background = `
                                    repeating-radial-gradient(
                                        circle at 50% 50%,
                                        ${{this.getRandomNightmareColor()}} 0px,
                                        transparent 5px,
                                        ${{this.getRandomNightmareColor()}} 10px
                                    )
                                `;
                                element.style.opacity = nightmareIntensity * 0.5;
                                element.style.mixBlendMode = 'luminosity';
                                break;
                        }}
                        
                        setTimeout(() => {{
                            element.style.opacity = '0';
                            element.style.filter = '';
                            element.style.boxShadow = '';
                            element.style.transform = '';
                        }}, Math.random() * 400 + 200);
                    }}
                }});
            }}
            
            applyGlobalChaos(progress) {{
                // 全局混乱效果
                if (this.screenTear && Math.random() < progress * 0.2) {{
                    const tearHeight = Math.random() * 100 + 20;
                    const tearY = Math.random() * (this.height - tearHeight);
                    
                    this.video1Layer.style.clipPath = `polygon(0 0, 100% 0, 100% ${{tearY}}px, 0 ${{tearY}}px)`;
                    this.video2Layer.style.clipPath = `polygon(0 ${{tearY + tearHeight}}px, 100% ${{tearY + tearHeight}}px, 100% 100%, 0 100%)`;
                    
                    setTimeout(() => {{
                        this.video1Layer.style.clipPath = '';
                        this.video2Layer.style.clipPath = '';
                    }}, 150);
                }}
                
                // 随机颜色爆炸
                if (Math.random() < this.chaosLevel * 0.05) {{
                    const explosionElement = this.chaosElements[Math.floor(Math.random() * 10)];
                    explosionElement.style.left = '0px';
                    explosionElement.style.top = '0px';
                    explosionElement.style.width = this.width + 'px';
                    explosionElement.style.height = this.height + 'px';
                    explosionElement.style.background = `
                        radial-gradient(circle at ${{Math.random() * 100}}% ${{Math.random() * 100}}%,
                            ${{this.getRandomNightmareColor()}} 0%,
                            transparent 30%)
                    `;
                    explosionElement.style.opacity = this.colorMadness * 0.3;
                    explosionElement.style.mixBlendMode = 'screen';
                    
                    setTimeout(() => {{
                        explosionElement.style.opacity = '0';
                    }}, 100);
                }}
            }}
            
            getRandomNightmareColor() {{
                const nightmareColors = [
                    '#ff0000', '#00ff00', '#0000ff', '#ff00ff', '#ffff00', '#00ffff',
                    '#ff8000', '#8000ff', '#ff0080', '#80ff00', '#0080ff', '#ff8080',
                    '#80ff80', '#8080ff', '#ffffff', '#000000'
                ];
                return nightmareColors[Math.floor(Math.random() * nightmareColors.length)];
            }}
            
            applyDigitalMeltdown(progress) {{
                // 数字崩溃效果 - 数字像素融化和数据流失
                const meltdownIntensity = progress * this.visualIntensity;
                const digitalPulse = Math.sin(progress * Math.PI * 35) * meltdownIntensity * 0.6;
                const totalMeltdown = meltdownIntensity + digitalPulse;
                
                // 数字融化滤镜
                this.video1Layer.style.filter = `
                    blur(${{totalMeltdown * 6}}px)
                    contrast(${{1.5 + totalMeltdown * 2}})
                    brightness(${{0.7 + Math.abs(digitalPulse) * 0.8}})
                    saturate(${{0.4 + totalMeltdown * 1.2}})
                    hue-rotate(${{digitalPulse * 540}}deg)
                    drop-shadow(0 ${{totalMeltdown * 10}}px 0 #00ff00)
                    drop-shadow(${{totalMeltdown * 8}}px 0 0 #ff0080)
                `;
                
                this.video2Layer.style.filter = `
                    contrast(${{2.5 + totalMeltdown}})
                    brightness(${{1.3 + digitalPulse * 0.4}})
                    saturate(${{1.8 + totalMeltdown * 0.8}})
                    drop-shadow(0 ${{-totalMeltdown * 6}}px 0 #0080ff)
                `;
                
                // 数字像素融化效果
                this.chaosElements.forEach((element, index) => {{
                    if (Math.random() < totalMeltdown * 0.7) {{
                        const meltType = index % 8;
                        
                        switch(meltType) {{
                            case 0: // 像素滴落
                                const dropSize = Math.random() * 20 + 8;
                                element.style.left = (Math.random() * this.width) + 'px';
                                element.style.top = (Math.random() * this.height * 0.3) + 'px';
                                element.style.width = dropSize + 'px';
                                element.style.height = (Math.random() * 100 + 50) + 'px';
                                element.style.background = `
                                    linear-gradient(180deg,
                                        #00ff00 0%,
                                        #80ff00 30%,
                                        #ffff00 60%,
                                        transparent 100%)
                                `;
                                element.style.opacity = Math.random() * 0.9 + 0.4;
                                element.style.mixBlendMode = 'screen';
                                element.style.borderRadius = '0 0 50% 50%';
                                element.style.animation = 'pixelDrop 1.5s ease-in infinite';
                                break;
                                
                            case 1: // 数字流失
                                element.style.left = (Math.random() * this.width) + 'px';
                                element.style.top = (Math.random() * this.height) + 'px';
                                element.style.width = (Math.random() * 6 + 3) + 'px';
                                element.style.height = (Math.random() * 200 + 100) + 'px';
                                element.style.background = `
                                    repeating-linear-gradient(180deg,
                                        #ff0080 0px,
                                        #00ff80 4px,
                                        #8000ff 8px,
                                        transparent 12px)
                                `;
                                element.style.opacity = Math.random() * 0.8 + 0.3;
                                element.style.mixBlendMode = 'overlay';
                                element.style.animation = 'dataStream 2s linear infinite';
                                break;
                                
                            case 2: // 像素块融化
                                const blockSize = Math.random() * 60 + 30;
                                element.style.left = (Math.random() * this.width) + 'px';
                                element.style.top = (Math.random() * this.height) + 'px';
                                element.style.width = blockSize + 'px';
                                element.style.height = blockSize + 'px';
                                element.style.background = `
                                    conic-gradient(
                                        from ${{Math.random() * 360}}deg,
                                        #ff4000 0deg,
                                        #00ff40 90deg,
                                        #4000ff 180deg,
                                        #ff0040 270deg,
                                        #ff4000 360deg
                                    )
                                `;
                                element.style.opacity = Math.random() * 0.7 + 0.2;
                                element.style.mixBlendMode = 'color-dodge';
                                element.style.filter = `blur(${{Math.random() * 8}}px)`;
                                element.style.animation = 'pixelMelt 1s ease-out infinite';
                                break;
                                
                            case 3: // 数字撕裂
                                element.style.left = '0px';
                                element.style.top = (Math.random() * this.height) + 'px';
                                element.style.width = this.width + 'px';
                                element.style.height = (Math.random() * 3 + 1) + 'px';
                                element.style.background = `
                                    linear-gradient(90deg,
                                        #ffffff 0%,
                                        #ff0080 20%,
                                        #00ff80 40%,
                                        #8000ff 60%,
                                        #ff8000 80%,
                                        #ffffff 100%)
                                `;
                                element.style.opacity = '1';
                                element.style.mixBlendMode = 'difference';
                                element.style.animation = 'digitalTear 0.5s linear infinite';
                                break;
                                
                            case 4: // 像素爆炸
                                const explodeSize = Math.random() * 40 + 20;
                                element.style.left = (Math.random() * this.width) + 'px';
                                element.style.top = (Math.random() * this.height) + 'px';
                                element.style.width = explodeSize + 'px';
                                element.style.height = explodeSize + 'px';
                                element.style.borderRadius = '50%';
                                element.style.background = `
                                    radial-gradient(circle,
                                        #ffffff 0%,
                                        #ff0080 30%,
                                        #00ff80 60%,
                                        transparent 100%)
                                `;
                                element.style.opacity = Math.random() * 0.9 + 0.5;
                                element.style.mixBlendMode = 'screen';
                                element.style.boxShadow = `
                                    0 0 30px #ffffff,
                                    0 0 60px #ff0080,
                                    0 0 90px #00ff80
                                `;
                                element.style.animation = 'pixelExplode 0.8s ease-out infinite';
                                break;
                                
                            case 5: // 数字雪花
                                const snowSize = Math.random() * 12 + 6;
                                element.style.left = (Math.random() * this.width) + 'px';
                                element.style.top = '-20px';
                                element.style.width = snowSize + 'px';
                                element.style.height = snowSize + 'px';
                                element.style.backgroundColor = '#ffffff';
                                element.style.opacity = Math.random() * 0.8 + 0.4;
                                element.style.mixBlendMode = 'screen';
                                element.style.borderRadius = '50%';
                                element.style.boxShadow = '0 0 15px #ffffff';
                                element.style.animation = 'digitalSnow 3s linear infinite';
                                break;
                                
                            case 6: // 像素波纹
                                const rippleSize = Math.random() * 150 + 80;
                                element.style.left = (Math.random() * this.width) + 'px';
                                element.style.top = (Math.random() * this.height) + 'px';
                                element.style.width = rippleSize + 'px';
                                element.style.height = rippleSize + 'px';
                                element.style.borderRadius = '50%';
                                element.style.background = `
                                    repeating-radial-gradient(
                                        circle at 50% 50%,
                                        transparent 0px,
                                        #00ff80 8px,
                                        transparent 16px,
                                        #ff0080 24px,
                                        transparent 32px
                                    )
                                `;
                                element.style.opacity = Math.random() * 0.6 + 0.2;
                                element.style.mixBlendMode = 'overlay';
                                element.style.animation = 'pixelRipple 2s ease-out infinite';
                                break;
                                
                            case 7: // 数字碎片
                                const fragmentSize = Math.random() * 25 + 10;
                                element.style.left = (Math.random() * this.width) + 'px';
                                element.style.top = (Math.random() * this.height) + 'px';
                                element.style.width = fragmentSize + 'px';
                                element.style.height = fragmentSize + 'px';
                                element.style.background = `
                                    linear-gradient(${{Math.random() * 360}}deg,
                                        #ff0080 0%,
                                        #00ff80 50%,
                                        #8000ff 100%)
                                `;
                                element.style.opacity = Math.random() * 0.8 + 0.3;
                                element.style.mixBlendMode = 'hard-light';
                                element.style.transform = `rotate(${{Math.random() * 360}}deg)`;
                                element.style.animation = 'digitalFragment 1.2s ease-in-out infinite';
                                break;
                        }}
                        
                        setTimeout(() => {{
                            element.style.opacity = '0';
                            element.style.animation = '';
                            element.style.transform = '';
                            element.style.filter = '';
                            element.style.boxShadow = '';
                            element.style.borderRadius = '';
                        }}, Math.random() * 1000 + 500);
                    }}
                }});
                
                // 数字世界崩塌
                if (Math.random() < totalMeltdown * 0.3) {{
                    document.body.style.background = `
                        conic-gradient(
                            from ${{Math.random() * 360}}deg at 50% 50%,
                            #00ff00 0deg,
                            #ff0080 60deg,
                            #0080ff 120deg,
                            #ff8000 180deg,
                            #8000ff 240deg,
                            #ffff00 300deg,
                            #00ff00 360deg
                        )
                    `;
                    setTimeout(() => {{
                        document.body.style.background = '{background_color}';
                    }}, 120);
                }}
            }}
        }}
        
        // 添加动画样式
        const style = document.createElement('style');
        style.textContent = `
            // 量子衰变动画
            @keyframes quantumSpin {{
                from {{ transform: rotate(0deg) scale(1); }}
                to {{ transform: rotate(360deg) scale(1.2); }}
            }}
            
            @keyframes quantumPulse {{
                0% {{ transform: scale(1); opacity: 0.8; }}
                50% {{ transform: scale(1.5); opacity: 1; }}
                100% {{ transform: scale(1); opacity: 0.8; }}
            }}
            
            @keyframes quantumTunnel {{
                0% {{ transform: translateX(-100%) scaleY(1); }}
                50% {{ transform: translateX(0%) scaleY(2); }}
                100% {{ transform: translateX(100%) scaleY(1); }}
            }}
            
            @keyframes quantumWave {{
                0% {{ transform: scale(1) rotate(0deg); filter: hue-rotate(0deg); }}
                50% {{ transform: scale(1.3) rotate(180deg); filter: hue-rotate(180deg); }}
                100% {{ transform: scale(1) rotate(360deg); filter: hue-rotate(360deg); }}
            }}
            
            // 神经网络崩溃动画
            @keyframes neuronPulse {{
                0% {{ transform: scale(1); box-shadow: 0 0 25px #ff0040; }}
                100% {{ transform: scale(1.4); box-shadow: 0 0 50px #ff4080, 0 0 100px #ff0040; }}
            }}
            
            @keyframes synapseFlicker {{
                0%, 100% {{ opacity: 0.3; }}
                50% {{ opacity: 1; }}
            }}
            
            @keyframes matrixRotate {{
                from {{ transform: rotate(0deg); }}
                to {{ transform: rotate(360deg); }}
            }}
            
            @keyframes activationGlitch {{
                0% {{ transform: translateX(0px); }}
                25% {{ transform: translateX(-5px); }}
                50% {{ transform: translateX(5px); }}
                75% {{ transform: translateX(-3px); }}
                100% {{ transform: translateX(0px); }}
            }}
            
            @keyframes backpropSpin {{
                0% {{ transform: rotate(0deg) scale(1); }}
                50% {{ transform: rotate(180deg) scale(1.2); }}
                100% {{ transform: rotate(360deg) scale(1); }}
            }}
            
            @keyframes gradientExplode {{
                0% {{ transform: scale(1); }}
                50% {{ transform: scale(3); }}
                100% {{ transform: scale(1); }}
            }}
            
            @keyframes overfitCollapse {{
                0% {{ transform: scale(1) rotate(0deg); }}
                100% {{ transform: scale(0.5) rotate(180deg); }}
            }}
            
            // 数字崩溃动画
            @keyframes pixelDrop {{
                0% {{ transform: translateY(0px); opacity: 1; }}
                100% {{ transform: translateY(200px); opacity: 0; }}
            }}
            
            @keyframes dataStream {{
                0% {{ transform: translateY(-50px); opacity: 0; }}
                20% {{ opacity: 1; }}
                80% {{ opacity: 1; }}
                100% {{ transform: translateY(200px); opacity: 0; }}
            }}
            
            @keyframes pixelMelt {{
                0% {{ transform: scale(1) rotate(0deg); }}
                50% {{ transform: scale(1.2) rotate(180deg); }}
                100% {{ transform: scale(0.8) rotate(360deg); }}
            }}
            
            @keyframes digitalTear {{
                0% {{ transform: translateX(-100%); }}
                100% {{ transform: translateX(100%); }}
            }}
            
            @keyframes pixelExplode {{
                0% {{ transform: scale(0.5); opacity: 1; }}
                50% {{ transform: scale(2); opacity: 0.8; }}
                100% {{ transform: scale(0.2); opacity: 0; }}
            }}
            
            @keyframes digitalSnow {{
                0% {{ transform: translateY(-20px) rotate(0deg); opacity: 1; }}
                100% {{ transform: translateY(200px) rotate(360deg); opacity: 0; }}
            }}
            
            @keyframes pixelRipple {{
                0% {{ transform: scale(0.5); opacity: 0.8; }}
                50% {{ transform: scale(1.5); opacity: 0.4; }}
                100% {{ transform: scale(2); opacity: 0; }}
            }}
            
            @keyframes digitalFragment {{
                0% {{ transform: rotate(0deg) scale(1); opacity: 0.8; }}
                50% {{ transform: rotate(180deg) scale(1.3); opacity: 1; }}
                100% {{ transform: rotate(360deg) scale(0.7); opacity: 0.3; }}
            }}
        `;
        document.head.appendChild(style);
        
        // 初始化控制器
        window.enhancedGlitchController = new EnhancedGlitchController();
        // console.log('✅ EnhancedGlitchController initialized');
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
    "VideoGlitchArtEnhancedNode": VideoGlitchArtEnhancedNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoGlitchArtEnhancedNode": "Video Glitch Art Enhanced",
}
