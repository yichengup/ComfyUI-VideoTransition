"""
视频扭曲转场节点 - OpenCV像素位移版本（批处理优化）
两个视频片段之间的扭曲转场效果，支持多种扭曲模式
基于OpenCV remap函数实现真正的像素位移扭曲
"""

import torch
import numpy as np
import cv2
import time
from comfy.comfy_types.node_typing import ComfyNodeABC, InputTypeDict, IO


class VideoWarpTransitionNode(ComfyNodeABC):
    """视频扭曲转场 - 基于OpenCV像素位移的扭曲转场（批处理优化版）"""
    
    CATEGORY = "VideoTransition"
    
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "video1": (IO.IMAGE,),  # 第一个视频
                "video2": (IO.IMAGE,),  # 第二个视频
                "warp_type": ([
                    "swirl",         # 旋涡扭曲
                    "squeeze_h",     # 水平挤压
                    "squeeze_v",     # 垂直挤压
                    "liquid",        # 液体扭曲
                    "wave",          # 波浪扭曲
                ],),
                "total_frames": (IO.INT, {"default": 60, "min": 4, "max": 300}),
                "fps": (IO.INT, {"default": 30, "min": 15, "max": 60}),
            },
            "optional": {
                "warp_intensity": (IO.FLOAT, {"default": 0.5, "min": 0.1, "max": 2.0}),
                "warp_speed": (IO.FLOAT, {"default": 1.0, "min": 0.1, "max": 3.0}),
                "max_scale": (IO.FLOAT, {"default": 1.3, "min": 1.0, "max": 3.0, "step": 0.1}),
                "scale_recovery": (IO.BOOLEAN, {"default": True}),
                "use_gpu": (IO.BOOLEAN, {"default": False}),
                "batch_size": (IO.INT, {"default": 5, "min": 1, "max": 20}),
                "background_color": (IO.STRING, {"default": "#000000"}),
                "width": (IO.INT, {"default": 640, "min": 320, "max": 1920}),
                "height": (IO.INT, {"default": 640, "min": 240, "max": 1080}),
            }
        }
    
    RETURN_TYPES = (IO.IMAGE,)
    RETURN_NAMES = ("video",)
    FUNCTION = "generate_warp_transition"
    
    async def generate_warp_transition(
        self, 
        video1, 
        video2,
        warp_type,
        total_frames, 
        fps,
        warp_intensity=0.5,
        warp_speed=1.0,
        max_scale=1.3,
        scale_recovery=True,
        use_gpu=False,
        batch_size=5,
        background_color="#000000",
        width=640,
        height=640
    ):
        """生成视频扭曲转场效果 - 基于OpenCV像素位移的批处理优化版本"""
        
        start_time = time.time()
        
        warp_names = {
            "swirl": "旋涡扭曲", 
            "squeeze_h": "水平挤压",
            "squeeze_v": "垂直挤压",
            "liquid": "液体扭曲",
            "wave": "波浪扭曲",
        }
        
        print(f"🎬 开始生成视频扭曲转场（OpenCV像素位移）...")
        print(f"   扭曲类型: {warp_names.get(warp_type, warp_type)}")
        print(f"   扭曲强度: {warp_intensity}")
        print(f"   扭曲速度: {warp_speed}")
        print(f"   最大缩放: {max_scale}x")
        print(f"   缩放恢复: {'启用' if scale_recovery else '禁用'}")
        print(f"   批处理大小: {batch_size}")
        
        # 提取视频帧
        frames1 = self._extract_video_frames(video1)
        frames2 = self._extract_video_frames(video2)
        
        print(f"📹 视频1: {len(frames1)}帧, 视频2: {len(frames2)}帧")
        print(f"🎞️ 转场帧数: {total_frames}")
        
        # 解析背景颜色
        bg_color_bgr = self._parse_background_color(background_color)
        
        # 批处理渲染
        print("🎬 开始批处理渲染...")
        render_start = time.time()
        
        output_frames = []
        
        for batch_start in range(0, total_frames, batch_size):
            batch_end = min(batch_start + batch_size, total_frames)
            batch_indices = list(range(batch_start, batch_end))
            
            print(f"📦 处理批次 {batch_start//batch_size + 1}/{(total_frames-1)//batch_size + 1}: 帧 {batch_start+1}-{batch_end}")
            
            # 批处理：一次处理多个帧
            batch_frames = self._process_batch_opencv(
                frames1, frames2, batch_indices, total_frames, warp_type, 
                warp_intensity, warp_speed, max_scale, scale_recovery, bg_color_bgr, width, height
            )
            
            # 立即添加到结果中，避免内存积累
            output_frames.extend(batch_frames)
            
            # 打印批次进度
            batch_progress = (batch_end / total_frames) * 100
            print(f"✅ 批次完成，总进度: {batch_progress:.1f}%")
        
        render_time = time.time() - render_start
        print(f"✅ 批处理渲染完成，耗时: {render_time:.2f}秒")
        
        # 转换为tensor
        video_tensor = self._frames_to_tensor(output_frames)
        
        total_time = time.time() - start_time
        print(f"✅ 扭曲转场完成，输出: {video_tensor.shape}")
        print(f"⏱️ 总耗时: {total_time:.2f}秒")
        print(f"🚀 平均每帧: {total_time/total_frames*1000:.1f}ms")
        
        return (video_tensor,)
    
    def _process_batch_opencv(self, frames1, frames2, batch_indices, total_frames, warp_type, 
                             warp_intensity, warp_speed, max_scale, scale_recovery, bg_color_bgr, width, height):
        """使用OpenCV进行批处理扭曲渲染"""
        batch_frames = []
        
        for i in batch_indices:
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
            
            # 调整到目标尺寸
            frame1_resized = cv2.resize(frame1_data, (width, height), interpolation=cv2.INTER_AREA)
            frame2_resized = cv2.resize(frame2_data, (width, height), interpolation=cv2.INTER_AREA)
            
            # 应用扭曲效果 - 剪映风格
            # 第一个视频：扭曲但不缩放
            frame1_warped = self._apply_warp_effect(frame1_resized, warp_type, warp_intensity, warp_speed, progress, 1.0 - progress, is_first_video=True, max_scale=max_scale, scale_recovery=scale_recovery)
            
            # 第二个视频：反向扭曲 + 缩放恢复
            frame2_warped = self._apply_warp_effect(frame2_resized, warp_type, warp_intensity, warp_speed, progress, progress, is_first_video=False, max_scale=max_scale, scale_recovery=scale_recovery)
            
            # 确保两个帧的尺寸一致
            if frame1_warped.shape != frame2_warped.shape:
                frame2_warped = cv2.resize(frame2_warped, (frame1_warped.shape[1], frame1_warped.shape[0]), interpolation=cv2.INTER_AREA)
            
            # 混合两个扭曲后的帧 - 确保无黑色背景的平滑过渡
            alpha1, alpha2 = self._calculate_blend_weights(progress)
            
            # 使用addWeighted进行混合，确保无黑色背景
            # alpha1和alpha2已经在_calculate_blend_weights中确保总和为1
            blended = cv2.addWeighted(frame1_warped, alpha1, frame2_warped, alpha2, 0)
            
            batch_frames.append(blended)
        
        return batch_frames
    
    def _apply_warp_effect(self, image, warp_type, intensity, speed, progress, alpha, is_first_video=True, max_scale=1.3, scale_recovery=True):
        """应用扭曲效果到单帧图像 - 剪映风格"""
        height, width = image.shape[:2]
        
        # 创建网格
        y, x = np.indices((height, width), dtype=np.float32)
        
        # 计算时间因子
        time_factor = progress * speed * np.pi * 2
        
        # 根据扭曲类型和视频类型计算位移
        if warp_type == "swirl":
            x_offset, y_offset = self._calculate_swirl_displacement(x, y, width, height, intensity, time_factor, progress, is_first_video)
        elif warp_type == "squeeze_h":
            x_offset, y_offset = self._calculate_squeeze_displacement(x, y, width, height, intensity, time_factor, progress, "horizontal", is_first_video)
        elif warp_type == "squeeze_v":
            x_offset, y_offset = self._calculate_squeeze_displacement(x, y, width, height, intensity, time_factor, progress, "vertical", is_first_video)
        elif warp_type == "liquid":
            x_offset, y_offset = self._calculate_liquid_displacement(x, y, width, height, intensity, time_factor, progress, is_first_video)
        elif warp_type == "wave":
            x_offset, y_offset = self._calculate_wave_displacement(x, y, width, height, intensity, time_factor, progress, is_first_video)
        else:
            x_offset, y_offset = np.zeros_like(x), np.zeros_like(y)
        
        # 计算新的坐标
        x_new = x + x_offset
        y_new = y + y_offset
        
        # 确保坐标在有效范围内
        x_new = np.clip(x_new, 0, width-1)
        y_new = np.clip(y_new, 0, height-1)
        
        # 应用扭曲
        warped = cv2.remap(image, x_new, y_new, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        
        # 第二个视频：应用缩放恢复效果
        if not is_first_video and scale_recovery:
            warped = self._apply_scale_recovery(warped, progress, width, height, max_scale)
        
        # 应用透明度
        if alpha < 1.0:
            warped = (warped * alpha).astype(np.uint8)
        
        return warped
    
    def _apply_scale_recovery(self, image, progress, width, height, max_scale=1.3):
        """应用缩放恢复效果 - 使用裁切方法实现第二个视频的缩放效果"""
        h, w = image.shape[:2]
        
        # 计算缩放比例：progress=0时最大(max_scale)，progress=1时正常(1.0x)
        scale = max_scale - progress * (max_scale - 1.0)
        
        # 确保缩放值在合理范围内
        scale = max(0.8, min(scale, max_scale))
        
        # 如果不需要缩放，直接返回原图
        if abs(scale - 1.0) < 0.01:
            return image
        
        # 使用裁切方法：从中心裁切出缩放后的区域
        # 当scale > 1时，相当于放大（裁切更小的区域）
        # 当scale < 1时，相当于缩小（裁切更大的区域）
        
        # 计算裁切区域的大小（与缩放效果相反）
        crop_scale = 1.0 / scale
        crop_w = int(w * crop_scale)
        crop_h = int(h * crop_scale)
        
        # 确保裁切尺寸不超过原图
        crop_w = min(crop_w, w)
        crop_h = min(crop_h, h)
        
        # 计算裁切起始位置（居中裁切）
        start_x = (w - crop_w) // 2
        start_y = (h - crop_h) // 2
        
        # 确保裁切位置在有效范围内
        start_x = max(0, min(start_x, w - crop_w))
        start_y = max(0, min(start_y, h - crop_h))
        
        # 裁切图像
        cropped = image[start_y:start_y + crop_h, start_x:start_x + crop_w]
        
        # 将裁切后的图像缩放回原始尺寸
        result = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_AREA)
        
        return result
    
    def _calculate_blend_weights(self, progress):
        """计算叠化权重 - 确保无黑色背景的平滑过渡"""
        # 使用平滑的S型曲线，确保alpha总和始终为1
        # 避免出现黑色背景，保持视频内容的连续性
        
        # 确保progress在有效范围内
        progress = max(0.0, min(progress, 1.0))
        
        # 使用更平滑的过渡曲线，避免突然的切换
        # 第一个视频：从1.0平滑过渡到0.0
        alpha1 = 1.0 - progress
        
        # 第二个视频：从0.0平滑过渡到1.0
        alpha2 = progress
        
        # 应用S型曲线让过渡更自然
        # 使用smoothstep函数：3t² - 2t³
        def smoothstep(t):
            return 3 * t * t - 2 * t * t * t
        
        # 对alpha2应用smoothstep，让过渡更自然
        alpha2 = smoothstep(alpha2)
        alpha1 = 1.0 - alpha2
        
        # 确保权重在合理范围内，并且总和为1
        alpha1 = max(0.0, min(alpha1, 1.0))
        alpha2 = max(0.0, min(alpha2, 1.0))
        
        # 最终确保alpha总和为1，避免黑色背景
        total_alpha = alpha1 + alpha2
        if total_alpha > 0:
            alpha1 = alpha1 / total_alpha
            alpha2 = alpha2 / total_alpha
        
        return alpha1, alpha2
    
    def _calculate_swirl_displacement(self, x, y, width, height, intensity, time_factor, progress, is_first_video=True):
        """计算旋涡扭曲的位移 - 剪映风格"""
        center_x = width / 2
        center_y = height / 2
        max_radius = np.sqrt(center_x**2 + center_y**2)
        swirl_intensity = intensity * 2
        
        # 计算到中心的距离和角度
        dx = x - center_x
        dy = y - center_y
        distance = np.sqrt(dx**2 + dy**2)
        angle = np.arctan2(dy, dx)
        
        # 计算影响因子
        influence = np.where(distance > 0, 
                           np.clip(1.0 - distance / max_radius, 0, 1), 0)
        influence = influence * influence * (3 - 2 * influence)
        
        if is_first_video:
            # 第一个视频：逐渐增强的扭曲
            rotation_factor = progress
            rotation_direction = 1
        else:
            # 第二个视频：反向扭曲恢复
            rotation_factor = 1.0 - progress
            rotation_direction = -1
        
        # 计算扭转角度
        twist_angle = (distance / max_radius) * swirl_intensity * np.pi * influence * rotation_factor * rotation_direction
        
        # 计算新坐标
        cos_theta = np.cos(twist_angle)
        sin_theta = np.sin(twist_angle)
        x_new = center_x + dx * cos_theta - dy * sin_theta
        y_new = center_y + dx * sin_theta + dy * cos_theta
        
        # 计算位移
        x_offset = x_new - x
        y_offset = y_new - y
        
        return x_offset, y_offset
    
    def _calculate_squeeze_displacement(self, x, y, width, height, intensity, time_factor, progress, direction, is_first_video=True):
        """计算挤压扭曲的位移 - 剪映风格"""
        if is_first_video:
            # 第一个视频：逐渐增强的挤压
            squeeze_intensity = intensity * progress * 58
        else:
            # 第二个视频：反向挤压恢复
            squeeze_intensity = intensity * (1.0 - progress) * 58
        
        if direction == "horizontal":
            center_x = width / 2
            squeeze = np.sin((x - center_x) / width * np.pi * 3) * squeeze_intensity
            x_offset = squeeze
            y_offset = np.zeros_like(x)
        else:  # vertical
            center_y = height / 2
            squeeze = np.sin((y - center_y) / height * np.pi * 3) * squeeze_intensity
            x_offset = np.zeros_like(x)
            y_offset = squeeze
        
        return x_offset, y_offset
    
    def _calculate_liquid_displacement(self, x, y, width, height, intensity, time_factor, progress, is_first_video=True):
        """计算液体扭曲的位移 - 剪映风格"""
        if is_first_video:
            # 第一个视频：逐渐增强的液体效果
            liquid_intensity = intensity * progress * 30
        else:
            # 第二个视频：反向液体恢复
            liquid_intensity = intensity * (1.0 - progress) * 30
        
        # 多方向波动
        wave1 = np.sin(x * 0.02 + time_factor) * liquid_intensity
        wave2 = np.cos(y * 0.02 + time_factor * 0.7) * liquid_intensity * 0.8
        
        # 添加额外的液体波动层
        wave3 = np.sin(x * 0.03 + time_factor * 1.3) * liquid_intensity * 0.4
        wave4 = np.cos(y * 0.03 + time_factor * 0.5) * liquid_intensity * 0.3
        
        x_offset = wave1 + wave3
        y_offset = wave2 + wave4
        
        return x_offset, y_offset
    
    def _calculate_wave_displacement(self, x, y, width, height, intensity, time_factor, progress, is_first_video=True):
        """计算波浪扭曲的位移 - 剪映风格"""
        if is_first_video:
            # 第一个视频：逐渐增强的波浪
            wave_intensity = intensity * progress * 40
        else:
            # 第二个视频：反向波浪恢复
            wave_intensity = intensity * (1.0 - progress) * 40
        
        # 波浪形扭曲
        wave1 = np.sin(x * 0.03 + time_factor) * wave_intensity
        wave2 = np.sin(x * 0.05 + time_factor * 1.5) * wave_intensity * 0.6
        
        # 添加垂直方向的波浪
        wave3 = np.sin(y * 0.02 + time_factor * 0.8) * wave_intensity * 0.4
        
        x_offset = wave3  # 垂直波浪影响x方向
        y_offset = wave1 + wave2  # 水平波浪影响y方向
        
        return x_offset, y_offset
    
    def _parse_background_color(self, color_str):
        """解析背景颜色字符串为BGR格式"""
        try:
            # 移除#号
            if color_str.startswith('#'):
                color_str = color_str[1:]
            
            # 解析RGB
            if len(color_str) == 6:
                r = int(color_str[0:2], 16)
                g = int(color_str[2:4], 16)
                b = int(color_str[4:6], 16)
            elif len(color_str) == 3:
                r = int(color_str[0] * 2, 16)
                g = int(color_str[1] * 2, 16)
                b = int(color_str[2] * 2, 16)
            else:
                # 默认黑色
                r, g, b = 0, 0, 0
            
            # 返回BGR格式（OpenCV使用）
            return (b, g, r)
        except:
            # 解析失败时返回黑色
            return (0, 0, 0)
    
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
    "VideoWarpTransitionNode": VideoWarpTransitionNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoWarpTransitionNode": "Video Warp Transition",
}
