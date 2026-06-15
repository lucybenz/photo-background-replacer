# 图片抠像换背景工具

这是一个图片人像抠像和背景替换工具。输入一张人物照片和一张背景图，输出一张合成 PNG。项目包含命令行版本和 WinForms UI 版本。

## 功能

- 单张人物照片抠像
- 替换为指定背景图片
- 输出 PNG 成品
- ONNX Runtime DirectML 推理，优先使用 Windows 独立显卡
- 支持命令行批处理
- 支持 WinForms 图形界面

## 环境要求

- Windows 10/11
- .NET 8 SDK
- 支持 DirectML 的显卡和驱动

## 模型

请下载 RVM ONNX 模型，并放到：

```text
models\rvm_mobilenetv3_fp32.onnx
```

推荐模型：

```text
https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_fp32.onnx
```

模型文件较大，默认不提交到 GitHub。

## 启动 UI

在项目目录运行：

```powershell
.\start_ui.cmd
```

或者：

```powershell
dotnet run --project .\PhotoBackgroundReplacerUI
```

## 命令行用法

```powershell
.\run_replace.cmd "人物照片路径" "背景图路径"
```

完整命令：

```powershell
dotnet run --project .\PhotoBackgroundReplacer -- --input person.jpg --background bg.jpg --output result.png
```

## 参数

- `--input`：人物照片路径
- `--background`：背景图片路径
- `--output`：输出 PNG 路径，不填写时保存到 `outputs`
- `--downsample`：RVM 下采样比例，默认 `0.125`
- `--provider`：`auto`、`dml` 或 `cpu`，默认 `auto`

## 使用建议

- 输入照片越清晰，头发和边缘效果越好。
- 背景图片会自动按人物照片尺寸 cover 裁切。
- 输出成品建议使用 PNG，便于保留高质量边缘。

## 开源协议

本项目使用 MIT License。第三方依赖、RVM 模型和相关算法请遵守其原始项目许可证。
