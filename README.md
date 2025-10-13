# ComfyUI-VideoTransition

基于 Chromium 无头浏览器和python实现的视频转场效果插件

通过 Playwright 驱动 Chromium 浏览器在无头模式下运行，利用现代 Web 技术（HTML5/CSS3/Canvas）渲染专业级视频转场效果。

## ✨ 核心特点

- 🌐 **浏览器引擎渲染** - Chromium 引擎驱动，充分利用 GPU 硬件加速
- 🎨 **Web 技术栈** - 使用 HTML5/CSS3/JavaScript 实现复杂动画效果
- 🐍 **纯 Python 控制** - 无需 Node.js，通过 Playwright 自动化框架控制浏览器
- ⚡ **高性能渲染** - 批处理优化，内存占用低，渲染速度快
- 🔧 **稳定可靠** - Microsoft Playwright + Google Chromium，跨平台支持

## 📦 安装

### 1. 安装依赖

```bash
cd custom_nodes/ComfyUI-VideoTransition
pip install -r requirements.txt
```

### 2. 安装 Chromium 浏览器

**Windows:**
```cmd
playwright install chromium
```

**Linux:**
```bash

playwright install chromium

# 安装系统依赖
playwright install-deps chromium
```


## 🎬 转场效果

本插件采用**混合技术架构**，根据效果复杂度选择最佳实现方案：

### 🌐 Chromium 引擎渲染（复杂 3D 效果）

#### 1. 立方体转场 ⭐
- **技术**：CSS 3D Transform + GPU 加速
- **效果**：水平/垂直立方体旋转
- **优势**：真实 3D 透视，硬件加速渲染
- **参数**：旋转角度、立方体大小、透视强度

#### 2. 立方体旋转展示
- **技术**：WebGL 3D 渲染
- **效果**：多视频立方体旋转动画
- **优势**：支持多面体、自由旋转

### ⚡ 纯 Python 实现（高性能）

#### 3. 视频叠化转场
- **技术**：NumPy 数组混合 + OpenCV
- **效果**：直接叠化、闪黑、闪白、叠加溶解、色散叠化
- **优势**：速度快、内存占用低
- **参数**：支持自定义颜色

#### 4. 3D 翻转转场
- **技术**：批处理优化 + 透视变换
- **效果**：水平/垂直/斜向翻转
- **优势**：纯 Python 实现真实 3D 效果

#### 5. 扭曲转场
- **技术**：OpenCV remap 像素位移
- **效果**：旋涡、水平/垂直挤压、液体、波浪
- **优势**：真实像素级扭曲效果
- **参数**：扭曲强度、速度、缩放

#### 6. 翻页转场
- **技术**：批处理优化 + 物理模拟
- **效果**：模拟真实书页翻动
- **优势**：逼真物理效果，多方向翻页

#### 7. 眨眼转场 ⭐
- **技术**：NumPy 遮罩 + cv2 高斯模糊
- **效果**：弧形眼睑模拟眨眼
- **优势**：性能提升 50%+，内存降低 60%+
- **参数**：眨眼速度、模糊强度、眼睑弧度

### 📊 技术选型说明

| 实现方式 | 适用场景 | 性能 | 优势 |
|---------|---------|-----|-----|
| **Chromium 渲染** | 复杂 3D 效果 | 中 | GPU 加速、CSS3 能力 |
| **纯 Python** | 2D/简单效果 | 高 | 速度快、内存低、稳定 |

## 🚀 使用方法

1. 在 ComfyUI 中找到 `VideoTransition` 分类
2. 选择需要的转场效果节点
3. 连接输入图片/视频
4. 调整参数
5. 运行工作流

## 💡 技术优势

### 混合技术架构 - 最佳方案选择

**🎯 设计理念：** 根据效果复杂度和性能需求，智能选择实现方式

#### 🌐 Chromium 引擎（用于复杂 3D 效果）
- ✅ **CSS3 3D 变换** - 复杂立体效果轻松实现
- ✅ **GPU 硬件加速** - 自动利用显卡渲染
- ✅ **开发效率高** - HTML/CSS 代码，所见即所得
- ✅ **跨平台一致** - 统一的 Chromium 引擎

**适用场景：** 立方体旋转等复杂 3D 透视效果

#### ⚡ 纯 Python（用于高性能需求）
- ✅ **性能极致** - 速度快 50%+，内存低 60%+
- ✅ **稳定可靠** - 无浏览器依赖，兼容性好
- ✅ **精确控制** - 像素级处理，效果可控
- ✅ **资源占用低** - 适合批量处理

**适用场景：** 叠化、翻页、扭曲等 2D/简单效果

#### 技术架构对比
```
Chromium 方案: Python → Playwright → Chromium → 视频帧
纯 Python 方案: Python → NumPy/OpenCV → 视频帧
```

> 💡 **优势：** 既能实现复杂 3D 效果，又保证高性能处理

**扩展潜力：**

Playwright 支持驱动三大浏览器引擎：
- **Chromium** - 当前使用，最快最稳定 ⭐
- **Firefox** - Mozilla 引擎，不同渲染特性
- **WebKit** - Safari 引擎，苹果生态

> 💡 未来可能扩展：选择不同浏览器引擎以获得特定渲染效果或兼容性

## ⚠️ 常见问题

**Q: 安装失败？**  
A: 使用镜像：`playwright install chromium --mirror=https://registry.npmmirror.com`

**Q: 渲染慢？**  
A: 降低 fps 或分辨率，减少 duration

**Q: 内存占用高？**  
A: 确保使用 `with` 语句关闭渲染器

## 📄 许可证

MIT License
