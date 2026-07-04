# VoxCPM2 声音克隆系统

基于 OpenBMB **VoxCPM2**（2B 参数，48kHz）的 Mac 本地声音克隆工具。在 M4 Pro 上运行，通过 Apple MPS 加速，无需 GPU。

## 核心功能

| 功能 | 描述 |
|------|------|
| **可控克隆** | 一段参考音频 + 文本 → 克隆该人声说出任意内容（无需参考文本） |
| **终极克隆** | 参考音频 + 参考文本 → 最高保真度声音克隆 |
| **声音设计** | 纯文字描述声音风格 → 生成对应风格语音（无需参考音频） |
| **纯 TTS** | 默认音色文本转语音 |
| **自动分段** | 长文本自动切段（≤50 字/段），独立生成后拼接，避免自回归累积漂移 |
| **自动降噪** | 参考音频自动 ffmpeg anlmdn 降噪 + VoxCPM2 内置降噪器 |
| **自动播放** | 生成完成后自动调用系统 `afplay` 播放语音 |

## 项目结构

```
voice-clone/
├── clone_voice.py             # 推理脚本（唯一入口）
├── models/openbmb/VoxCPM2/    # 模型权重 (4.58 GB)
│   ├── model.safetensors      # 4.3G 主模型
│   ├── audiovae.pth           # 359M 音频编解码器
│   ├── tokenizer/             # 分词器
│   └── config/                # 模型配置
├── ref_audio/                 # 参考音频存放目录
├── output/                    # 生成音频输出目录
├── .venv/                     # Python 3.12 虚拟环境
├── pyproject.toml             # 项目配置
└── README.md
```

## 环境要求

- **硬件**: Mac M 系列芯片（M4 Pro 测试通过），建议 ≥16GB 内存
- **系统**: macOS Sonoma / Sequoia
- **软件**: `ffmpeg`, `sox`, `uv`（Python 3.12 管理）
- **依赖**: torch + voxcpm + soundfile + modelscope

## 快速使用

### 激活环境

```bash
cd voice-clone/
source .venv/bin/activate
```

### 纯 TTS（默认音色）

```bash
python clone_voice.py \
  --text "你好，欢迎体验语音合成系统。" \
  --output output/tts.wav
```

### 声音设计（免参考音频）

用文字描述目标声音，无需任何参考音频：

```bash
python clone_voice.py \
  --text "(英国管家，沉稳冷静，语速平缓)先生您好，您的下午茶已经准备好了。" \
  --output output/design.wav
```

描述词格式：`(风格描述)正文内容`，放在文本最前面。

### 可控克隆（免参考文本）

用一段参考音频克隆人声，合成时无需提供参考文本：

```bash
python clone_voice.py \
  --text "要合成的文本内容" \
  --ref-audio ref_audio/ref.wav \
  --output output/clone.wav
```

### 终极克隆（最高保真度）

需要参考音频 + 参考文本，声音相似度最高：

```bash
python clone_voice.py \
  --text "要合成的文本内容" \
  --prompt-audio ref_audio/ref.wav \
  --prompt-text "参考音频的原文内容" \
  --output output/ultimate_clone.wav
```

## 完整参数说明

| 参数 | 简写 | 默认值 | 说明 |
|------|------|--------|------|
| `--text` | `-t` | **必填** | 要合成的文本（支持声音描述词前置） |
| `--output` | `-o` | `output.wav` | 输出音频文件路径 |
| `--device` | | `auto` | 运行设备：`auto` / `mps` / `cuda` / `cpu` |
| **模式互斥组** | | | |
| `--ref-audio` | `-ra` | 无 | 参考音频路径（可控克隆模式） |
| `--prompt-audio` | `-pa` | 无 | 提示音频路径（终极克隆模式） |
| `--prompt-text` | `-pt` | 无 | 提示文本（终极克隆必填） |
| **生成参数** | | | |
| `--cfg` | | `2.0` | CFG guidance scale（1.0-3.0，越高越像参考音色） |
| `--steps` | | `10` | 推理步数（4-30，越大越精细但越慢） |
| `--seed` | | 随机 | 随机种子，固定后可复现结果 |
| **功能开关** | | | |
| `--no-denoiser` | | — | 禁用 VoxCPM2 内置降噪器 |
| `--no-optimize` | | — | 禁用 torch.compile 优化（降内存但更慢） |
| `--no-play` | | — | 生成后不自动播放（默认自动播放） |

## 进阶功能

### 长文本自动分段

文本超过 50 字时自动按标点符号切段，每段独立生成后无缝拼接。这是为了解决自回归模型的累积漂移问题（长文本后半段出现嗡嗡声）。

可通过环境变量自定义分段字数：

```bash
VOXCPM_MAX_CHARS=100 python clone_voice.py \
  --text "较长的文本内容..." \
  --ref-audio ref_audio/ref.wav \
  --output output/segmented.wav
```

### 参考音频自动降噪

无论输入音频是 M4A、MP3 还是其他格式，脚本自动用 `ffmpeg anlmdn` 降噪并转为 16kHz/mono WAV 后输入模型。

### 生成后自动播放

默认调用系统 `afplay` 播放生成的 WAV 文件（阻塞，播完才继续）。

关闭：加 `--no-play` 参数。

### 随机种子复现

固定 seed 后同一文本 + 同一音频组合可复现完全相同的结果：

```bash
python clone_voice.py \
  --text "测试文本" \
  --ref-audio ref_audio/ref.wav \
  --output output/reproducible.wav \
  --seed 42
```

## 技术规格

| 项目 | 值 |
|------|-----|
| **基础模型** | OpenBMB VoxCPM2（2B 参数） |
| **输出格式** | 48kHz / 16-bit / mono WAV |
| **输入格式** | 自动转 16kHz / mono（ffmpeg） |
| **支持语言** | 30 种语言 + 9 种中文方言 |
| **许可证** | Apache-2.0 |
| **模型大小** | 4.58 GB |

## 性能数据（M4 Pro 24GB）

| 模式 | RTF（实时率） | 5 秒语音耗时 |
|------|:-----------:|:-----------:|
| 纯 TTS | ~1.3x | ~6.5s |
| 声音设计 | ~1.6x | ~8s |
| 声音克隆（10 steps） | ~4.4x | ~22s |
| 声音克隆（20 steps + 降噪） | ~7x | ~35s |
| **模型首次加载** | — | **~20s**（含 warmup ~5s） |

## 常见问题

**Q: 第一次运行报错？**
首次有 ~5 秒 warmup（torch.compile 回退到 eager 模式），属正常现象。

**Q: 生成的声音不像参考音频？**
- 增加 `--steps`（建议 15-20）
- 调整 `--cfg`（建议 2.0-2.5）
- 确保参考音频干净（无背景噪音）
- 推荐参考音频长度 15-30 秒

**Q: 长文本后半段有嗡嗡声？**
脚本已自动按 50 字分段，如果仍有问题，减小 `VOXCPM_MAX_CHARS` 环境变量（如设为 30）。

**Q: MPS 报错 / 显存不足？**
加 `--no-optimize` 禁用 torch.compile。MPS 会自动使用 float32（bfloat16 在扩散循环中不稳定）。
