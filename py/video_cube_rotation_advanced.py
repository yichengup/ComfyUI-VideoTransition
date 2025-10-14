"""
多视频立方体旋转效果节点 - 高级混合架构版本
真正的3D投影计算 + OpenCV合成
"""

import torch
import cv2
import numpy as np
from PIL import Image
from comfy.comfy_types.node_typing import ComfyNodeABC, InputTypeDict, IO


class VideoCubeRotationAdvancedNode(ComfyNodeABC):
    """多视频立方体旋转效果节点 - 高级混合架构版本"""
    
    CATEGORY = "VideoTransition"
    
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
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
                "video_front": (IO.IMAGE,),   # 前面视频
                "video_back": (IO.IMAGE,),    # 后面视频
                "video_left": (IO.IMAGE,),    # 左面视频
                "video_right": (IO.IMAGE,),   # 右面视频
                "video_top": (IO.IMAGE,),     # 上面视频
                "video_bottom": (IO.IMAGE,),  # 下面视频
                "background_color": (IO.STRING, {"default": "#1a1a2e"}),
                "width": (IO.INT, {"default": 640, "min": 640, "max": 3840}),
                "height": (IO.INT, {"default": 640, "min": 360, "max": 2160}),
                "cube_size": (IO.FLOAT, {"default": 0.6, "min": 0.3, "max": 1.0}),
                "perspective": (IO.FLOAT, {"default": 1000.0, "min": 500.0, "max": 3000.0}),
            }
        }
    
    RETURN_TYPES = (IO.IMAGE,)
    RETURN_NAMES = ("video",)
    FUNCTION = "generate_video_cube_rotation"
    
    def __init__(self):
        # 立方体的8个顶点（3D坐标）
        self.cube_vertices = np.array([
            [-1, -1, -1],  # 0: 左后下
            [ 1, -1, -1],  # 1: 右后下
            [ 1,  1, -1],  # 2: 右前下
            [-1,  1, -1],  # 3: 左前下
            [-1, -1,  1],  # 4: 左后上
            [ 1, -1,  1],  # 5: 右后上
            [ 1,  1,  1],  # 6: 右前上
            [-1,  1,  1],  # 7: 左前上
        ], dtype=np.float32)
        
        # 立方体的6个面（每个面4个顶点索引）
        self.cube_faces = [
            [0, 1, 2, 3],  # 前面 (front)
            [4, 7, 6, 5],  # 后面 (back)
            [1, 5, 6, 2],  # 右面 (right)
            [4, 0, 3, 7],  # 左面 (left)
            [3, 2, 6, 7],  # 上面 (top)
            [4, 5, 1, 0],  # 下面 (bottom)
        ]
        
        # 面名称映射
        self.face_names = ['front', 'back', 'right', 'left', 'top', 'bottom']
    
    async def generate_video_cube_rotation(
        self, 
        rotation_style, 
        rotation_speed, 
        duration, 
        fps,
        video_front=None,
        video_back=None,
        video_left=None,
        video_right=None,
        video_top=None,
        video_bottom=None,
        background_color="#1a1a2e",
        width=1920,
        height=1080,
        cube_size=0.6,
        perspective=1000.0
    ):
        """生成多视频立方体旋转效果 - 高级混合架构"""
        
        print(f"Starting cube rotation: {rotation_style}, {int(duration * fps)} frames")
        
        # 处理6个面的视频数据
        video_frames_data = self._process_cube_videos(
            video_front, video_back, video_left, video_right, video_top, video_bottom
        )
        
        # 计算总帧数
        total_frames = int(duration * fps)
        
        # 统计视频面数
        face_count = sum(1 for frames in video_frames_data if len(frames) > 0)
        print(f"Video cube faces: {face_count}/6 faces provided, generating {total_frames} frames")
        
        # 纯Python渲染（不使用Playwright）
        video_tensor = self._render_with_opencv(
            video_frames_data, rotation_style, rotation_speed,
            total_frames, width, height, cube_size, perspective, background_color
        )
        
        return (video_tensor,)
    
    def _process_cube_videos(self, video_front, video_back, video_left, video_right, video_top, video_bottom):
        """处理6个面的视频数据，转换为numpy数组"""
        video_frames_data = []
        videos = [video_front, video_back, video_left, video_right, video_top, video_bottom]
        
        for i, video_tensor in enumerate(videos):
            if video_tensor is not None:
                if video_tensor.dim() == 4:
                    batch_size = video_tensor.shape[0]
                    frames = []
                    for j in range(batch_size):
                        frame_tensor = video_tensor[j]  # [H, W, C]
                        frame_np = self._tensor_to_numpy(frame_tensor)
                        frames.append(frame_np)
                    
                    video_frames_data.append(frames)
                else:
                    raise ValueError(f"{self.face_names[i]} video tensor must be 4D format [B, H, W, C], current: {video_tensor.shape}")
            else:
                default_frame = np.zeros((512, 512, 3), dtype=np.uint8)
                video_frames_data.append([default_frame])
        
        return video_frames_data
    
    def _tensor_to_numpy(self, tensor):
        """将tensor转换为numpy数组"""
        if tensor.dim() == 4:
            tensor = tensor[0]
        
        array = tensor.cpu().numpy()
        
        if array.dtype in [np.float32, np.float64]:
            array = (array * 255).clip(0, 255).astype(np.uint8)
        elif array.dtype != np.uint8:
            array = array.astype(np.uint8)
        
        return array
    
    def _render_with_opencv(
        self, 
        video_frames_data, 
        rotation_style, 
        rotation_speed,
        total_frames, 
        width, 
        height, 
        cube_size, 
        perspective, 
        background_color
    ):
        """使用OpenCV进行纯Python渲染"""
        
        # 解析背景颜色
        bg_color = self._parse_background_color(background_color)
        
        frames = []
        
        for i in range(total_frames):
            progress = i / max(total_frames - 1, 1)
            
            # 创建背景
            frame = np.full((height, width, 3), bg_color, dtype=np.uint8)
            
            # 计算旋转矩阵
            rotation_matrix = self._calculate_rotation_matrix(rotation_style, progress, rotation_speed)
            
            # 渲染立方体
            self._render_cube(frame, video_frames_data, rotation_matrix, progress, cube_size, perspective, width, height)
            
            # 转换为tensor
            frame_tensor = torch.from_numpy(frame.astype(np.float32) / 255.0)
            frames.append(frame_tensor)
            
            # 显示进度 - 每20帧或完成时显示
            if (i + 1) % 20 == 0 or i == total_frames - 1:
                progress_pct = (i + 1) / total_frames * 100
                print(f"Rendering frames {i + 1}/{total_frames} ({progress_pct:.1f}%)")
        
        # 堆叠为视频tensor
        video_tensor = torch.stack(frames, dim=0)
        print(f"Cube rotation completed: {video_tensor.shape}")
        
        return video_tensor
    
    def _parse_background_color(self, color_str):
        """解析背景颜色字符串"""
        if color_str.startswith('#'):
            # 十六进制颜色
            hex_color = color_str[1:]
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            return [b, g, r]  # OpenCV使用BGR格式
        else:
            # 默认黑色
            return [0, 0, 0]
    
    def _calculate_rotation_matrix(self, rotation_style, progress, rotation_speed):
        """计算旋转矩阵"""
        angle = progress * 360 * rotation_speed
        
        if rotation_style == 'x_axis':
            return self._rotation_matrix_x(np.radians(angle))
        elif rotation_style == 'y_axis':
            return self._rotation_matrix_y(np.radians(angle))
        elif rotation_style == 'z_axis':
            return self._rotation_matrix_z(np.radians(angle))
        elif rotation_style == 'diagonal':
            rx = self._rotation_matrix_x(np.radians(angle * 0.7))
            ry = self._rotation_matrix_y(np.radians(angle))
            return np.dot(ry, rx)
        elif rotation_style == 'random_spin':
            rx = self._rotation_matrix_x(np.radians(angle * 0.6))
            ry = self._rotation_matrix_y(np.radians(angle * 1.0))
            rz = self._rotation_matrix_z(np.radians(angle * 0.3))
            return np.dot(rz, np.dot(ry, rx))
        elif rotation_style == 'cube_unfold':
            if progress < 0.5:
                return self._rotation_matrix_y(np.radians(progress * 2 * 180))
            else:
                unfold_progress = (progress - 0.5) * 2
                return self._rotation_matrix_y(np.radians(180 + unfold_progress * 180))
        else:
            return self._rotation_matrix_y(np.radians(angle))
    
    def _rotation_matrix_x(self, angle):
        """绕X轴旋转矩阵"""
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        return np.array([
            [1, 0, 0],
            [0, cos_a, -sin_a],
            [0, sin_a, cos_a]
        ], dtype=np.float32)
    
    def _rotation_matrix_y(self, angle):
        """绕Y轴旋转矩阵"""
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        return np.array([
            [cos_a, 0, sin_a],
            [0, 1, 0],
            [-sin_a, 0, cos_a]
        ], dtype=np.float32)
    
    def _rotation_matrix_z(self, angle):
        """绕Z轴旋转矩阵"""
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        return np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ], dtype=np.float32)
    
    def _render_cube(self, frame, video_frames_data, rotation_matrix, progress, cube_size, perspective, width, height):
        """渲染立方体"""
        
        # 缩放立方体
        scale = cube_size * min(width, height) / 4
        vertices = self.cube_vertices * scale
        
        # 应用旋转
        rotated_vertices = np.dot(vertices, rotation_matrix.T)
        
        # 透视投影
        projected_vertices = self._perspective_projection(rotated_vertices, perspective, width, height)
        
        # 渲染每个面
        for face_idx, face_vertex_indices in enumerate(self.cube_faces):
            if face_idx < len(video_frames_data) and video_frames_data[face_idx]:
                self._render_face(frame, face_vertex_indices, projected_vertices, 
                                video_frames_data[face_idx], progress, face_idx)
    
    def _perspective_projection(self, vertices, perspective, width, height):
        """透视投影"""
        # 将3D点投影到2D屏幕
        projected = np.zeros((len(vertices), 2), dtype=np.float32)
        
        for i, vertex in enumerate(vertices):
            if vertex[2] + perspective > 0:  # 避免除零
                # 透视投影
                x = (vertex[0] * perspective) / (vertex[2] + perspective)
                y = (vertex[1] * perspective) / (vertex[2] + perspective)
                
                # 转换到屏幕坐标
                projected[i][0] = x + width / 2
                projected[i][1] = y + height / 2
        
        return projected
    
    def _render_face(self, frame, face_vertex_indices, projected_vertices, face_frames, progress, face_idx):
        """渲染立方体的一个面"""
        
        # 获取面的4个顶点
        face_vertices = projected_vertices[face_vertex_indices]
        
        # 检查面是否可见（简单的背面剔除）
        if not self._is_face_visible(face_vertices):
            return
        
        # 计算当前帧索引
        total_frames = len(face_frames)
        frame_index = int(progress * (total_frames - 1))
        frame_index = max(0, min(frame_index, total_frames - 1))
        
        # 获取当前帧
        current_frame = face_frames[frame_index]
        
        # 创建变换矩阵
        src_points = np.array([
            [0, 0],
            [current_frame.shape[1], 0],
            [current_frame.shape[1], current_frame.shape[0]],
            [0, current_frame.shape[0]]
        ], dtype=np.float32)
        
        dst_points = face_vertices.astype(np.float32)
        
        # 计算透视变换矩阵
        try:
            transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            
            # 应用透视变换
            warped = cv2.warpPerspective(current_frame, transform_matrix, (frame.shape[1], frame.shape[0]))
            
            # 创建掩码
            mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            cv2.fillPoly(mask, [dst_points.astype(np.int32)], 255)
            
            # 合成到帧上
            frame[mask > 0] = warped[mask > 0]
            
        except cv2.error:
            # 如果变换失败，跳过这个面
            pass
    
    def _is_face_visible(self, face_vertices):
        """简单的背面剔除"""
        # 计算面的法向量（简化版本）
        v1 = face_vertices[1] - face_vertices[0]
        v2 = face_vertices[2] - face_vertices[0]
        
        # 叉积计算法向量
        normal_z = v1[0] * v2[1] - v1[1] * v2[0]
        
        # 如果法向量指向屏幕外，则面不可见
        return normal_z > 0


# 注册节点
NODE_CLASS_MAPPINGS = {
    "VideoCubeRotationAdvancedNode": VideoCubeRotationAdvancedNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoCubeRotationAdvancedNode": "Multi-video cube rotation",
}

