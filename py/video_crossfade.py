"""
视频透明叠化转场节点
两个视频片段之间的简单透明叠化转场效果
"""

import torch
import numpy as np
import cv2
from comfy.comfy_types.node_typing import ComfyNodeABC, InputTypeDict, IO


class VideoCrossfadeNode(ComfyNodeABC):
    """视频透明叠化转场 - 两个视频之间的多种转场模式"""
    
    CATEGORY = "VideoTransition"
    
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "video1": (IO.IMAGE,),  # 第一个视频
                "video2": (IO.IMAGE,),  # 第二个视频
                "transition_mode": ([
                    "crossfade",           # 直接叠化
                    "fade_to_black",       # 闪黑
                    "fade_to_white",       # 闪白
                    "fade_to_custom",      # 闪自定义色
                    "additive_dissolve",   # 叠加
                    "chromatic_dissolve",  # 色散叠化
                ],),
                "total_frames": (IO.INT, {"default": 30, "min": 1, "max": 300}),
                "fps": (IO.INT, {"default": 30, "min": 15, "max": 60}),
            },
            "optional": {
                "background_color": (IO.STRING, {"default": "#FF0000"}),  # 用于 fade_to_custom 模式，默认红色
                "width": (IO.INT, {"default": 640, "min": 64, "max": 3840}),
                "height": (IO.INT, {"default": 640, "min": 64, "max": 2160}),
            }
        }
    
    RETURN_TYPES = (IO.IMAGE,)
    RETURN_NAMES = ("video",)
    FUNCTION = "generate_crossfade"
    
    async def generate_crossfade(
        self, 
        video1, 
        video2,
        transition_mode,
        total_frames, 
        fps,
        background_color="#000000",
        width=640,
        height=640
    ):
        """生成视频转场效果"""
        
        # 提取视频帧
        frames1 = self._extract_video_frames(video1)
        frames2 = self._extract_video_frames(video2)
        
        print(f"Video crossfade transition: {len(frames1)} + {len(frames2)} frames -> {total_frames} frames")
        print(f"Mode: {transition_mode}")
        
        # 解析背景颜色
        bg_color_bgr = self._parse_background_color(background_color)
        
        # 使用OpenCV直接合成（简单高效）
        output_frames = []
        
        for i in range(total_frames):
            progress = i / (total_frames - 1) if total_frames > 1 else 0
            
            # 获取对应的帧 - 根据进度获取
            frame1_idx = min(int(progress * (len(frames1) - 1)), len(frames1) - 1)
            frame2_idx = min(int(progress * (len(frames2) - 1)), len(frames2) - 1)
            
            frame1 = frames1[frame1_idx]
            frame2 = frames2[frame2_idx]
            
            # 根据转场模式进行不同的混合
            if transition_mode == "crossfade":
                # 直接叠化
                blended_frame = self._blend_frames_with_background(
                    frame1, frame2, progress, bg_color_bgr, width, height
                )
            elif transition_mode == "fade_to_black":
                # 闪黑：video1 → 黑色 → video2
                blended_frame = self._fade_through_color(
                    frame1, frame2, progress, (0, 0, 0), width, height
                )
            elif transition_mode == "fade_to_white":
                # 闪白：video1 → 白色 → video2
                blended_frame = self._fade_through_color(
                    frame1, frame2, progress, (255, 255, 255), width, height
                )
            elif transition_mode == "fade_to_custom":
                # 闪自定义色：video1 → 自定义颜色 → video2
                blended_frame = self._fade_through_color(
                    frame1, frame2, progress, bg_color_bgr, width, height
                )
            elif transition_mode == "additive_dissolve":
                # 叠加：使用加法混合
                blended_frame = self._additive_blend(
                    frame1, frame2, progress, width, height
                )
            elif transition_mode == "chromatic_dissolve":
                # 色散叠化：RGB通道错位
                blended_frame = self._chromatic_blend(
                    frame1, frame2, progress, width, height
                )
            else:
                # 默认使用直接叠化
                blended_frame = self._blend_frames_with_background(
                    frame1, frame2, progress, bg_color_bgr, width, height
                )
            
            output_frames.append(blended_frame)
            
            # 显示进度 - 每20帧或完成时显示
            if (i + 1) % 20 == 0 or i == total_frames - 1:
                progress_percent = (i + 1) / total_frames * 100
                print(f"Processing frames {i+1}/{total_frames} ({progress_percent:.1f}%)")
        
        # 转换为tensor
        video_tensor = self._frames_to_tensor(output_frames)
        
        print(f"Crossfade transition completed: {video_tensor.shape}")
        return (video_tensor,)
    
    def _extract_video_frames(self, video_tensor):
        """从视频tensor中提取帧"""
        frames = []
        
        # 视频tensor格式: [B, H, W, C]
        if video_tensor.dim() == 4:
            batch_size = video_tensor.shape[0]
            for i in range(batch_size):
                frame = video_tensor[i]  # [H, W, C]
                frames.append(frame)
        else:
            # 单帧情况
            frames.append(video_tensor)
        
        return frames
    
    def _parse_background_color(self, color_str):
        """解析背景颜色字符串为BGR元组"""
        if color_str.startswith('#'):
            # 移除#号
            hex_color = color_str[1:]
            # 转换为RGB
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            # 返回BGR格式（OpenCV使用BGR）
            return (b, g, r)
        else:
            # 默认黑色
            return (0, 0, 0)
    
    def _fade_through_color(self, frame1, frame2, progress, color_bgr, width, height):
        """闪色转场：frame1 → 纯色 → frame2"""
        # 转换为numpy数组
        if isinstance(frame1, torch.Tensor):
            frame1 = frame1.cpu().numpy()
        if isinstance(frame2, torch.Tensor):
            frame2 = frame2.cpu().numpy()
        
        # 确保数据类型正确
        if frame1.dtype == np.float32 or frame1.dtype == np.float64:
            frame1 = (frame1 * 255).clip(0, 255).astype(np.uint8)
        if frame2.dtype == np.float32 or frame2.dtype == np.float64:
            frame2 = (frame2 * 255).clip(0, 255).astype(np.uint8)
        
        # 调整到目标尺寸
        frame1_resized = cv2.resize(frame1, (width, height), interpolation=cv2.INTER_AREA)
        frame2_resized = cv2.resize(frame2, (width, height), interpolation=cv2.INTER_AREA)
        
        # 创建纯色画布
        color_canvas = np.full((height, width, 3), color_bgr, dtype=np.uint8)
        
        # 分两个阶段：
        # 阶段1（0 -> 0.5）：frame1 淡出到纯色
        # 阶段2（0.5 -> 1.0）：纯色淡入到 frame2
        if progress < 0.5:
            # 阶段1：frame1 → 纯色
            fade_progress = progress * 2  # 映射到 0-1
            alpha1 = 1.0 - fade_progress
            alpha2 = fade_progress
            blended = cv2.addWeighted(frame1_resized, alpha1, color_canvas, alpha2, 0)
        else:
            # 阶段2：纯色 → frame2
            fade_progress = (progress - 0.5) * 2  # 映射到 0-1
            alpha1 = 1.0 - fade_progress
            alpha2 = fade_progress
            blended = cv2.addWeighted(color_canvas, alpha1, frame2_resized, alpha2, 0)
        
        return blended
    
    def _blend_frames_with_background(self, frame1, frame2, progress, bg_color_bgr, width, height):
        """使用OpenCV合成两帧，包含背景色处理"""
        # 转换为numpy数组
        if isinstance(frame1, torch.Tensor):
            frame1 = frame1.cpu().numpy()
        if isinstance(frame2, torch.Tensor):
            frame2 = frame2.cpu().numpy()
        
        # 确保数据类型正确
        if frame1.dtype == np.float32 or frame1.dtype == np.float64:
            frame1 = (frame1 * 255).clip(0, 255).astype(np.uint8)
        if frame2.dtype == np.float32 or frame2.dtype == np.float64:
            frame2 = (frame2 * 255).clip(0, 255).astype(np.uint8)
        
        # 调整到目标尺寸
        frame1_resized = cv2.resize(frame1, (width, height), interpolation=cv2.INTER_AREA)
        frame2_resized = cv2.resize(frame2, (width, height), interpolation=cv2.INTER_AREA)
        
        # 透明叠化: frame1逐渐淡出，frame2逐渐淡入
        alpha1 = 1.0 - progress  # frame1的透明度
        alpha2 = progress        # frame2的透明度
        
        # 使用OpenCV的addWeighted进行透明叠化
        blended = cv2.addWeighted(frame1_resized, alpha1, frame2_resized, alpha2, 0)
        
        # 创建背景色画布
        background_canvas = np.full((height, width, 3), bg_color_bgr, dtype=np.uint8)
        
        # 如果输入帧有Alpha通道，进行Alpha合成
        if len(blended.shape) == 3 and blended.shape[2] == 4:  # RGBA
            alpha_channel = blended[:, :, 3] / 255.0
            rgb_channels = blended[:, :, :3]
            
            # Alpha合成到背景
            for c in range(3):
                background_canvas[:, :, c] = (
                    background_canvas[:, :, c] * (1 - alpha_channel) + 
                    rgb_channels[:, :, c] * alpha_channel
                ).astype(np.uint8)
        else:  # RGB，直接使用混合结果
            background_canvas = blended
        
        return background_canvas
    
    def _additive_blend(self, frame1, frame2, progress, width, height):
        """叠加混合：两个视频直接相加，产生更亮的叠加效果"""
        # 转换为numpy数组
        if isinstance(frame1, torch.Tensor):
            frame1 = frame1.cpu().numpy()
        if isinstance(frame2, torch.Tensor):
            frame2 = frame2.cpu().numpy()
        
        # 确保数据类型正确
        if frame1.dtype == np.float32 or frame1.dtype == np.float64:
            frame1 = (frame1 * 255).clip(0, 255).astype(np.uint8)
        if frame2.dtype == np.float32 or frame2.dtype == np.float64:
            frame2 = (frame2 * 255).clip(0, 255).astype(np.uint8)
        
        # 调整到目标尺寸
        frame1_resized = cv2.resize(frame1, (width, height), interpolation=cv2.INTER_AREA)
        frame2_resized = cv2.resize(frame2, (width, height), interpolation=cv2.INTER_AREA)
        
        # 🎯 真正的叠加：使用Screen混合模式或增强的加法混合
        # Screen模式公式: 1 - (1-A) * (1-B)
        # 效果：两个视频相加但不会过度曝光，产生发光叠加效果
        
        # 方案：在转场中间时，两个视频都保持较高的可见度
        # 使用曲线让中间更亮
        if progress < 0.5:
            # 前半段：frame1保持满亮度，frame2逐渐增强
            alpha1 = 1.0
            alpha2 = progress * 2  # 0 -> 1
        else:
            # 后半段：frame1逐渐减弱，frame2保持满亮度
            alpha1 = (1.0 - progress) * 2  # 1 -> 0
            alpha2 = 1.0
        
        # 先转换为float并归一化到0-1
        frame1_norm = frame1_resized.astype(np.float32) / 255.0
        frame2_norm = frame2_resized.astype(np.float32) / 255.0
        
        # 🔥 使用Screen混合模式：1 - (1-A) * (1-B)
        # 这样两个亮的区域叠加会更亮，但不会完全过曝
        screen_blend = 1.0 - (1.0 - frame1_norm * alpha1) * (1.0 - frame2_norm * alpha2)
        
        # 转换回0-255
        blended = (screen_blend * 255.0).clip(0, 255).astype(np.uint8)
        
        return blended
    
    def _chromatic_blend(self, frame1, frame2, progress, width, height):
        """色散叠化：RGB通道错位混合，产生色散/色差效果"""
        # 转换为numpy数组
        if isinstance(frame1, torch.Tensor):
            frame1 = frame1.cpu().numpy()
        if isinstance(frame2, torch.Tensor):
            frame2 = frame2.cpu().numpy()
        
        # 确保数据类型正确
        if frame1.dtype == np.float32 or frame1.dtype == np.float64:
            frame1 = (frame1 * 255).clip(0, 255).astype(np.uint8)
        if frame2.dtype == np.float32 or frame2.dtype == np.float64:
            frame2 = (frame2 * 255).clip(0, 255).astype(np.uint8)
        
        # 调整到目标尺寸
        frame1_resized = cv2.resize(frame1, (width, height), interpolation=cv2.INTER_AREA)
        frame2_resized = cv2.resize(frame2, (width, height), interpolation=cv2.INTER_AREA)
        
        # 计算通道偏移量（在转场中间最大）
        # 使用sin曲线使偏移平滑
        max_offset = 10  # 最大偏移像素
        offset_intensity = np.sin(progress * np.pi)  # 0 -> 1 -> 0
        offset = int(max_offset * offset_intensity)
        
        # 创建输出图像
        blended = np.zeros_like(frame1_resized)
        
        # 计算混合权重
        alpha1 = 1.0 - progress
        alpha2 = progress
        
        # 分别处理RGB三个通道，每个通道施加不同的偏移
        # R通道：向右偏移
        if offset > 0:
            # frame1的R通道向右偏移
            blended[:, offset:, 2] = (
                frame1_resized[:, :-offset, 2].astype(np.float32) * alpha1 +
                frame2_resized[:, offset:, 2].astype(np.float32) * alpha2
            ).astype(np.uint8)
            blended[:, :offset, 2] = (
                frame1_resized[:, :offset, 2].astype(np.float32) * alpha1 +
                frame2_resized[:, :offset, 2].astype(np.float32) * alpha2
            ).astype(np.uint8)
        else:
            blended[:, :, 2] = (
                frame1_resized[:, :, 2].astype(np.float32) * alpha1 +
                frame2_resized[:, :, 2].astype(np.float32) * alpha2
            ).astype(np.uint8)
        
        # G通道：不偏移（中心）
        blended[:, :, 1] = (
            frame1_resized[:, :, 1].astype(np.float32) * alpha1 +
            frame2_resized[:, :, 1].astype(np.float32) * alpha2
        ).astype(np.uint8)
        
        # B通道：向左偏移
        if offset > 0:
            # frame1的B通道向左偏移
            blended[:, :-offset, 0] = (
                frame1_resized[:, offset:, 0].astype(np.float32) * alpha1 +
                frame2_resized[:, :-offset, 0].astype(np.float32) * alpha2
            ).astype(np.uint8)
            blended[:, -offset:, 0] = (
                frame1_resized[:, -offset:, 0].astype(np.float32) * alpha1 +
                frame2_resized[:, -offset:, 0].astype(np.float32) * alpha2
            ).astype(np.uint8)
        else:
            blended[:, :, 0] = (
                frame1_resized[:, :, 0].astype(np.float32) * alpha1 +
                frame2_resized[:, :, 0].astype(np.float32) * alpha2
            ).astype(np.uint8)
        
        return blended
    
    def _blend_frames(self, frame1, frame2, progress):
        """使用OpenCV合成两帧"""
        # 转换为numpy数组
        if isinstance(frame1, torch.Tensor):
            frame1 = frame1.cpu().numpy()
        if isinstance(frame2, torch.Tensor):
            frame2 = frame2.cpu().numpy()
        
        # 确保数据类型正确
        if frame1.dtype == np.float32 or frame1.dtype == np.float64:
            frame1 = (frame1 * 255).clip(0, 255).astype(np.uint8)
        if frame2.dtype == np.float32 or frame2.dtype == np.float64:
            frame2 = (frame2 * 255).clip(0, 255).astype(np.uint8)
        
        # 确保尺寸一致
        if frame1.shape != frame2.shape:
            frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))
        
        # 透明叠化: frame1逐渐淡出，frame2逐渐淡入
        alpha1 = 1.0 - progress  # frame1的透明度
        alpha2 = progress        # frame2的透明度
        
        # 使用OpenCV的addWeighted进行透明叠化
        blended = cv2.addWeighted(frame1, alpha1, frame2, alpha2, 0)
        
        return blended
    
    def _frames_to_tensor(self, frames):
        """将帧列表转换为视频tensor"""
        if not frames:
            return torch.zeros((1, 1080, 1920, 3), dtype=torch.float32)
        
        # 转换为tensor
        frame_tensors = []
        for frame in frames:
            if isinstance(frame, np.ndarray):
                # 转换为float32并归一化
                frame_tensor = torch.from_numpy(frame).float() / 255.0
                frame_tensors.append(frame_tensor)
            else:
                frame_tensors.append(frame)
        
        # 堆叠成视频tensor [B, H, W, C]
        video_tensor = torch.stack(frame_tensors, dim=0)
        
        return video_tensor


# 注册节点
NODE_CLASS_MAPPINGS = {
    "VideoCrossfadeNode": VideoCrossfadeNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoCrossfadeNode": "Video overlay",
}
