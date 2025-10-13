"""
立方体旋转效果节点 - Playwright版本
将6张图片贴在3D立方体上进行旋转展示
"""

import torch
import base64
import io
from PIL import Image
import numpy as np
from comfy.comfy_types.node_typing import ComfyNodeABC, InputTypeDict, IO
from .playwright_renderer import PlaywrightRenderer


class CubeRotationNode(ComfyNodeABC):
    """立方体旋转效果节点 - 3D立方体旋转展示"""
    
    CATEGORY = "VideoTransition"
    
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "images": (IO.IMAGE,),
                "rotation_style": ([
                    "x_axis",           # 绕X轴旋转
                    "y_axis",           # 绕Y轴旋转
                    "z_axis",           # 绕Z轴旋转
                    "diagonal",         # 对角线旋转
                    "random_spin",      # 随机旋转
                    "cube_unfold",      # 立方体展开
                ],),
                "rotation_speed": (IO.FLOAT, {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
                "duration": (IO.FLOAT, {"default": 5.0, "min": 1.0, "max": 30.0, "step": 0.1}),
                "fps": (IO.INT, {"default": 30, "min": 15, "max": 60}),
            },
            "optional": {
                "background_color": (IO.STRING, {"default": "#1a1a2e"}),
                "width": (IO.INT, {"default": 1920, "min": 640, "max": 3840}),
                "height": (IO.INT, {"default": 1080, "min": 360, "max": 2160}),
                "cube_size": (IO.FLOAT, {"default": 0.6, "min": 0.3, "max": 1.0}),
                "perspective": (IO.FLOAT, {"default": 1000.0, "min": 500.0, "max": 3000.0}),
            }
        }
    
    RETURN_TYPES = (IO.IMAGE,)
    RETURN_NAMES = ("video",)
    FUNCTION = "generate_cube_rotation"
    
    async def generate_cube_rotation(
        self, 
        images, 
        rotation_style, 
        rotation_speed, 
        duration, 
        fps,
        background_color="#1a1a2e",
        width=1920,
        height=1080,
        cube_size=0.6,
        perspective=1000.0
    ):
        """生成立方体旋转效果"""
        
        print(f"🎬 开始生成立方体旋转效果: {rotation_style}")
        
        # 将输入图片转换为base64列表
        image_base64_list = self._images_to_base64_list(images)
        
        # 确保至少有6张图片（立方体6个面）
        while len(image_base64_list) < 6:
            image_base64_list.extend(image_base64_list[:6-len(image_base64_list)])
        
        # 只使用前6张图片
        image_base64_list = image_base64_list[:6]
        
        # 使用Playwright渲染器
        async with PlaywrightRenderer() as renderer:
            # 生成HTML
            html_content = self._generate_html(
                image_base64_list, rotation_style, rotation_speed,
                background_color, cube_size, perspective, width, height
            )
            
            # 渲染视频帧
            video_tensor = await renderer.render_frames(html_content, duration, fps, width, height)
        
        return (video_tensor,)
    
    def _images_to_base64_list(self, images):
        """将图片tensor列表转换为base64列表"""
        base64_list = []
        
        for img_tensor in images:
            # 处理单张图片
            if img_tensor.dim() == 4:
                img_tensor = img_tensor[0]
            
            array = img_tensor.cpu().numpy()
            
            # 转换数据类型
            if array.dtype == np.float32 or array.dtype == np.float64:
                array = (array * 255).clip(0, 255).astype(np.uint8)
            
            # 转换为PIL图片
            image = Image.fromarray(array, mode='RGB')
            
            # 转换为base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            base64_list.append(f"data:image/png;base64,{img_str}")
        
        return base64_list
    
    def _generate_html(
        self, 
        image_base64_list, 
        rotation_style, 
        rotation_speed,
        background_color, 
        cube_size, 
        perspective,
        width,
        height
    ):
        """生成立方体旋转HTML模板"""
        
        # 计算立方体尺寸（使用偶数避免半像素问题）
        cube_width = int(600 * cube_size)
        if cube_width % 2 != 0:
            cube_width += 1
        
        # 精确的半宽度
        half_width = cube_width / 2
        
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
            display: flex;
            justify-content: center;
            align-items: center;
            overflow: hidden;
            perspective: {perspective}px;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }}
        
        .scene {{
            width: {cube_width}px;
            height: {cube_width}px;
            position: relative;
            transform-style: preserve-3d;
        }}
        
        .cube {{
            width: 100%;
            height: 100%;
            position: relative;
            transform-style: preserve-3d;
            transform: translateZ(-{half_width}px);
            will-change: transform;
        }}
        
        .cube-face {{
            position: absolute;
            width: {cube_width}px;
            height: {cube_width}px;
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            backface-visibility: hidden;
            background-size: 101%;
            box-shadow: inset 0 0 30px rgba(0,0,0,0.2);
            outline: 1px solid rgba(255, 255, 255, 0.1);
            outline-offset: -1px;
            image-rendering: -webkit-optimize-contrast;
            image-rendering: crisp-edges;
            transform-style: preserve-3d;
        }}
        
        .cube-face::before {{
            content: '';
            position: absolute;
            top: -1px;
            left: -1px;
            width: calc(100% + 2px);
            height: calc(100% + 2px);
            background: linear-gradient(
                135deg,
                rgba(255,255,255,0.08) 0%,
                transparent 50%,
                rgba(0,0,0,0.15) 100%
            );
            pointer-events: none;
        }}
        
        .front {{
            background-image: url('{image_base64_list[0]}');
            transform: rotateY(0deg) translateZ({half_width}px);
        }}
        
        .back {{
            background-image: url('{image_base64_list[1]}');
            transform: rotateY(180deg) translateZ({half_width}px);
        }}
        
        .right {{
            background-image: url('{image_base64_list[2]}');
            transform: rotateY(90deg) translateZ({half_width}px);
        }}
        
        .left {{
            background-image: url('{image_base64_list[3]}');
            transform: rotateY(-90deg) translateZ({half_width}px);
        }}
        
        .top {{
            background-image: url('{image_base64_list[4]}');
            transform: rotateX(90deg) translateZ({half_width}px);
        }}
        
        .bottom {{
            background-image: url('{image_base64_list[5]}');
            transform: rotateX(-90deg) translateZ({half_width}px);
        }}
        
        .light {{
            position: absolute;
            width: 200px;
            height: 200px;
            background: radial-gradient(circle, rgba(255,255,255,0.3), transparent);
            border-radius: 50%;
            pointer-events: none;
            filter: blur(40px);
        }}
    </style>
</head>
<body>
    <div class="light" id="light"></div>
    <div class="scene" id="scene">
        <div class="cube" id="cube">
            <div class="cube-face front"></div>
            <div class="cube-face back"></div>
            <div class="cube-face right"></div>
            <div class="cube-face left"></div>
            <div class="cube-face top"></div>
            <div class="cube-face bottom"></div>
        </div>
    </div>
    
    <script>
        // 配置
        window.cubeConfig = {{
            rotationStyle: "{rotation_style}",
            rotationSpeed: {rotation_speed},
            cubeSize: {cube_size}
        }};
        
        // 立方体旋转控制器
        class CubeController {{
            constructor() {{
                this.cube = document.getElementById('cube');
                this.light = document.getElementById('light');
                this.config = window.cubeConfig;
                this.ready = false;
            }}
            
            init() {{
                if (this.cube) {{
                    this.ready = true;
                    console.log('CubeController初始化完成:', this.config.rotationStyle);
                }}
            }}
            
            update(progress) {{
                if (!this.cube || !this.ready) return;
                
                const speed = this.config.rotationSpeed;
                const angle = progress * 360 * speed;
                
                let transform = '';
                
                switch(this.config.rotationStyle) {{
                    case 'x_axis':
                        transform = `translateZ(-{half_width}px) rotateX(${{angle}}deg)`;
                        break;
                        
                    case 'y_axis':
                        transform = `translateZ(-{half_width}px) rotateY(${{angle}}deg)`;
                        break;
                        
                    case 'z_axis':
                        transform = `translateZ(-{half_width}px) rotateZ(${{angle}}deg)`;
                        break;
                        
                    case 'diagonal':
                        transform = `translateZ(-{half_width}px) rotateX(${{angle}}deg) rotateY(${{angle}}deg)`;
                        break;
                        
                    case 'random_spin':
                        const angleX = angle * 0.7;
                        const angleY = angle * 1.3;
                        const angleZ = angle * 0.5;
                        transform = `translateZ(-{half_width}px) rotateX(${{angleX}}deg) rotateY(${{angleY}}deg) rotateZ(${{angleZ}}deg)`;
                        break;
                        
                    case 'cube_unfold':
                        const unfoldProgress = Math.sin(progress * Math.PI);
                        const rotateY = unfoldProgress * 180;
                        const scale = 1 + unfoldProgress * 0.5;
                        transform = `translateZ(-{half_width}px) rotateY(${{rotateY}}deg) scale(${{scale}})`;
                        break;
                        
                    default:
                        transform = `translateZ(-{half_width}px) rotateY(${{angle}}deg)`;
                }}
                
                this.cube.style.transform = transform;
                
                // 更新光照位置
                const lightX = Math.sin(progress * Math.PI * 2) * 300 + 150;
                const lightY = Math.cos(progress * Math.PI * 2) * 300 + 150;
                this.light.style.left = lightX + 'px';
                this.light.style.top = lightY + 'px';
                
                if (progress < 0.01 || progress > 0.99) {{
                    console.log(`Cube: progress=${{progress.toFixed(3)}}, angle=${{angle.toFixed(1)}}`);
                }}
            }}
        }}
        
        // 创建全局实例
        const cubeController = new CubeController();
        
        // 等待图片加载完成
        let loadedImages = 0;
        const totalImages = 6;
        
        document.querySelectorAll('.cube-face').forEach((face) => {{
            const bgImage = window.getComputedStyle(face).backgroundImage;
            if (bgImage && bgImage !== 'none') {{
                const img = new Image();
                img.onload = () => {{
                    loadedImages++;
                    if (loadedImages === totalImages) {{
                        cubeController.init();
                        window.effectReady = true;
                        console.log('✅ 所有立方体面加载完成，效果准备就绪');
                    }}
                }};
                img.onerror = () => {{
                    loadedImages++;
                    if (loadedImages === totalImages) {{
                        cubeController.init();
                        window.effectReady = true;
                        console.log('✅ 立方体面加载完成（部分失败），效果准备就绪');
                    }}
                }};
                // 从backgroundImage中提取URL
                const match = bgImage.match(/url\\(["']?([^"']*)["']?\\)/);
                if (match) {{
                    img.src = match[1];
                }}
            }} else {{
                loadedImages++;
            }}
        }});
        
        // 超时保护
        setTimeout(() => {{
            if (!window.effectReady) {{
                cubeController.init();
                window.effectReady = true;
                console.log('⚠️ 超时保护：强制标记效果就绪');
            }}
        }}, 3000);
        
        // 供渲染器调用的更新函数
        window.updateEffect = function(progress) {{
            cubeController.update(progress);
        }};
    </script>
</body>
</html>
"""


# 注册节点
NODE_CLASS_MAPPINGS = {
    "CubeRotationNode": CubeRotationNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CubeRotationNode": "Cube Rotation",
}
