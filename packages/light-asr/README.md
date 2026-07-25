# light-asr

独立的 ASR 能力包：从音视频中产出带时间戳的词列表（`list[Word]`）。

## 职责

- **音频提取**：任意视频/音频 → 16kHz 单声道 WAV（ffmpeg）
- **转录**：whisperX（默认，VAD + 强制对齐）或 whisper.cpp
- **对齐**：whisper.cpp 词时间戳的 wav2vec2 强制对齐修正
- **说话人标注**：pyannote diarization（可选）
- **checkpoint 序列化**：`light-asr-words.v1` 词列表 JSON 读写

whisperX / torch / pyannote 是运行时可选依赖（不在本包 `dependencies` 中），
仅在使用对应引擎时才需安装。

本包不知道"run 目录"等编排概念：调用方传入 `work_dir`，原始输出落在哪里由调用方决定。

## 独立接入示例

```python
from light_asr import AsrConfig, AsrEngine, extract_audio, transcribe

wav = extract_audio("input.mp4", "work/")          # → work/audio_asr.wav
words = transcribe(wav, AsrConfig(engine=AsrEngine.WHISPERX), work_dir="work/asr")
```

细粒度函数也可单独使用：`light_asr.whisperx.run`、`light_asr.whisper_cpp.transcribe`、
`light_asr.align.align_words`、`light_asr.diarize.run`、`light_asr.checkpoints.*`。
