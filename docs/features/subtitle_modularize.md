---
status: 已落地
---

Tips:进入这个需求前，先阅读principle.md，了解下我们的原则，有利于知道为什么这么做，并为怎么做打一个基础

## 字幕包模块化拆分
把light-subtitle进行模块化改造，拆分的模块分别提供以下能力：
- ASR/STT：核心价值是模态转换，即把声音变成文本
    - 多Provider支持（多维度：ASR、对齐、说话人分离）
    - 统一的输出Scheme

- ASR/STT Polish：核心价值是转录纠正、标点还原等
    - 统一的输出Scheme

- Subtile：核心价值是字幕拆分、字幕翻译、字幕排版（时空层面）
    - 未来目标支持多语种
    - 统一的输出Scheme

- Subtile Style：核心价值是字幕样式定制
    - 统一的输出Scheme

- CLI：提供统一的CLI接口，支持多种命令，可扩展，内部聚合多种能力

## 注意事项
- 这不是一个重构需求，不接受妥协，必要时重新实现
- 核心目标是拆了以后，用户可以独立接入，例如，一个已经有自己ASR方案的用户，只是想有字幕能力，要能最小侵入地接入Subtitle，把ASR结果做输入，就有字幕做输出

## 落地状态

拆分已完成并收尾（2026-07）。最终 8 个包结构与职责：

| 包 | 职责（一句话） |
|---|---|
| `light-models` | 纯数据契约（Word / Segment / SubtitleCue）与序列化，零逻辑依赖 |
| `light-text` | 文本工具：标点常量、时间码格式化、is_cjk |
| `light-core` | 运行时原语：logger 模块、ProgressCallback |
| `light-llm` | LLM 横切层：OpenAIClient / retry / json_extract / parallel / prompts / usage |
| `light-asr` | 独立 ASR 能力：`api.transcribe` 一次调用（extract_audio + whisperX/whisper.cpp + align + diarize + checkpoints），统一 `light-transcript.v1` 输出 |
| `light-asr-polish` | 独立 ASR 后处理：LLM 转录纠正 correct + 标点还原 restore_punct（prompts 包内置） |
| `light-subtitle` | 字幕能力：segment 断句 / plan 规划 / translate 翻译 / subtitle 布局对时 / style 样式 / export 导出（srt/vtt/ass + transcript.json），13 个 prompt 模板包内置 |
| `light-cli` | 编排层：17 步管线（`light-subtitle` / `light pipeline`）+ `light asr/polish/subtitle` 独立能力命令 |

独立接入路径已验证：外部 ASR 结果（任何 `light-transcript.v1` 格式的 transcript.json）可直接 `light subtitle -i transcript.json -o <dir> [--target-lang zh --bilingual]` 得到与管线一致的字幕产物；`light asr` / `light polish` 分别覆盖模态转换与转录后处理环节，三者以 transcript.json 为契约自由组合。repo 根 `prompts/` 目录已消亡，全部模板随包分发（importlib.resources 加载）。
