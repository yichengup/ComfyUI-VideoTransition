"""
Playwright渲染器基类
提供基础的HTML渲染和截图功能
"""

import os
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional

import io
import torch
import numpy as np
from PIL import Image

try:
    from playwright.async_api import async_playwright, Page, Browser
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    print("⚠️  Playwright未安装，请运行: pip install playwright && playwright install chromium")


class PlaywrightRenderer:
    """Playwright渲染器基类"""
    
    def __init__(self):
        self.playwright = None
        self.browser = None
        self.temp_dir = None
        
    async def __aenter__(self):
        """异步上下文管理器：进入"""
        if not PLAYWRIGHT_AVAILABLE:
            raise RuntimeError("Playwright未安装")
        
        # 创建临时目录
        self.temp_dir = Path(tempfile.mkdtemp(prefix="comfyui_playwright_"))
        
        # 启动Playwright
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=True,
            args=[
                '--disable-gpu',  # 先禁用GPU，确保稳定性
                '--disable-dev-shm-usage',
                '--disable-software-rasterizer',
                '--disable-extensions',
                '--disable-background-networking',
                '--disable-sync',
                '--disable-translate',
                '--hide-scrollbars',
                '--metrics-recording-only',
                '--mute-audio',
                '--no-first-run',
            ]
        )
        
        print("✅ Playwright启动成功")
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器：退出"""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        
    async def render_frames(
        self, 
        html_content: str, 
        duration: float, 
        fps: int,
        width: int = 1920,
        height: int = 1080
    ) -> torch.Tensor:
        """
        渲染HTML生成视频帧
        
        Args:
            html_content: HTML内容
            duration: 持续时间（秒）
            fps: 帧率
            width: 宽度
            height: 高度
            
        Returns:
            torch.Tensor: 视频帧 [B, H, W, C]
        """
        total_frames = int(duration * fps)
        frames = []
        
        # 创建页面
        page = await self.browser.new_page(viewport={'width': width, 'height': height})
        
        try:
            # 加载HTML
            await page.set_content(html_content, wait_until='domcontentloaded', timeout=30000)
            
            # 等待初始化
            await page.wait_for_function("window.effectReady === true", timeout=5000)
            
            print(f"📹 开始渲染 {total_frames} 帧...")
            
            # 渲染每一帧
            for i in range(total_frames):
                progress = i / max(total_frames - 1, 1)
                
                # 更新动画状态
                await page.evaluate(f"window.updateEffect({progress})")
                
                # 等待渲染完成
                await page.wait_for_timeout(16)  # 约等于60fps的一帧时间
                
                # 截图
                screenshot_bytes = await page.screenshot()
                
                # 转换为tensor
                image = Image.open(io.BytesIO(screenshot_bytes)).convert('RGB')
                array = np.array(image).astype(np.float32) / 255.0
                tensor = torch.from_numpy(array)
                frames.append(tensor)
                
                # 进度提示
                if (i + 1) % 10 == 0 or i == total_frames - 1:
                    print(f"   渲染进度: {i + 1}/{total_frames} ({(i + 1) / total_frames * 100:.1f}%)")
            
            # 堆叠为视频tensor [B, H, W, C]
            video_tensor = torch.stack(frames, dim=0)
            
            print(f"✅ 渲染完成！输出形状: {video_tensor.shape}")
            
            return video_tensor
            
        finally:
            await page.close()
    
    def save_image_to_temp(self, image_tensor: torch.Tensor, filename: str) -> str:
        """
        保存图片到临时目录
        
        Args:
            image_tensor: 图片tensor [H, W, C]
            filename: 文件名
            
        Returns:
            str: 文件路径
        """
        # 转换为numpy
        if image_tensor.dim() == 4:
            image_tensor = image_tensor[0]
        
        array = image_tensor.cpu().numpy()
        
        # 转换数据类型
        if array.dtype == np.float32 or array.dtype == np.float64:
            array = (array * 255).clip(0, 255).astype(np.uint8)
        
        # 保存图片
        image = Image.fromarray(array, mode='RGB')
        filepath = self.temp_dir / filename
        image.save(filepath)
        
        return str(filepath)


