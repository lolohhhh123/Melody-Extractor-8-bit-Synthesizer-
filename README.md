# Melody Extractor & 8-bit Synthesizer  
# 旋律提取器与 8-bit 合成器

[English](#english) | [中文](#中文)

---

## English <a name="english"></a>

### Overview
This tool extracts the main melody from an audio file (using RMVPE/CREPE/pYIN) and generates a **8-bit style multi-track WAV** (square wave, triangle, arpeggio, drums) with customizable parameters. It supports full-track extraction or only memorable segments based on audio structure analysis.

### Features
- Automatic BPM and beat detection (`librosa`)
- Melody extraction with RMVPE (primary) + CREPE/pYIN fallback
- 8-bit multi-track synthesis:
  - **Melody** (square wave)
  - **Bass** (triangle wave, octave down)
  - **Arpeggio** (pulse wave, 25% duty)
  - **Drums** (kick, snare, hi-hat, tom with different patterns)
- Adjustable per‑track volume, global time offset, and mute intervals
- JSON configuration save/load for easy parameter reuse
- Mixed output with original audio (stereo, high quality)

### Dependencies
Install required packages:
```bash
pip install -r requirements.txt

Core packages:

librosa==0.10.2.post1

MIDIUtil==1.2.1

numpy==1.24.3

scipy==1.10.1

soundfile==0.12.1

torch==1.11.0+cu115

torchaudio==0.11.0+cu115

torchvision==0.12.0+cu115

Optional (fallback pitch extraction):

crepe – pip install crepe (requires TensorFlow)

rmvpe

Usage
Basic command:

bash
python melody_extractor.py input.wav --full --render-wav --multitrack --drum-pattern rock --time-offset 0.05
Key Arguments
Argument	Description
--full	Extract melody from the entire song (default: only top memorable segments)
--render-wav	Generate WAV audio (sine wave if not --multitrack)
--multitrack	Enable 8‑bit multi‑track synthesis
--time-offset	Global time shift in seconds (positive = delay)
--melody-vol, --bass-vol, --arpeggio-vol, --drum-vol	Per‑track volume
--mute-intervals	Silence periods, e.g. "0,5;10,15" (seconds)
--drum-pattern	simple, rock, pop, or funk
--config	Load JSON config file
--save-config	Save current config to JSON
Output Files
_melody.mid – MIDI file of the extracted main melody

.wav – 8-bit synthesized audio (if --render-wav + --multitrack)

_mix.wav – Melody mixed with original audio (if --mix)

License
MIT

中文 <a name="中文"></a>
概述
本工具从音频文件中提取主旋律（使用 RMVPE/CREPE/pYIN），并生成 8-bit 风格的多轨 WAV（方波、三角波、琶音、鼓点），支持多种参数自定义。可提取全曲或仅根据音频结构分析提取高记忆片段。

特性
自动检测 BPM 和节拍（librosa）

主旋律提取：RMVPE（首选）+ CREPE/pYIN 备用

8-bit 多轨合成：

旋律（方波）

低音（三角波，降低八度）

琶音（脉冲波，占空比 25%）

鼓点（底鼓、军鼓、踩镲、嗵鼓，不同节奏型）

可调各轨音量、全局时间偏移、静音区间

JSON 配置保存/加载，方便复用参数

与原曲高质量混音输出（立体声，保留原采样率）

依赖
安装所需包：

bash
pip install -r requirements.txt

librosa==0.10.2.post1

MIDIUtil==1.2.1

numpy==1.24.3

scipy==1.10.1

soundfile==0.12.1

torch==1.11.0+cu115

torchaudio==0.11.0+cu115

torchvision==0.12.0+cu115

可选（备用音高提取）：

crepe – pip install crepe（需 TensorFlow）

rmvpe

使用方法
基础命令：

bash
python melody_extractor.py input.wav --full --render-wav --multitrack --drum-pattern rock --time-offset 0.05
主要参数
参数	说明
--full	提取全曲旋律（默认仅提取高记忆片段）
--render-wav	生成 WAV 音频（若不开启 --multitrack 则为正弦波）
--multitrack	启用 8-bit 多轨合成
--time-offset	全局时间偏移（秒），正数延迟，负数提前
--melody-vol, --bass-vol, --arpeggio-vol, --drum-vol	各轨音量
--mute-intervals	静音时间段，如 "0,5;10,15"（秒）
--drum-pattern	simple, rock, pop, funk
--config	加载 JSON 配置文件
--save-config	保存当前配置到 JSON
输出文件
_melody.mid – 提取的主旋律 MIDI 文件

.wav – 8-bit 合成音频（若开启 --render-wav + --multitrack）

_mix.wav – 旋律与原曲混音（若开启 --mix）

许可证
MIT
