# light-qc 转录对齐优化方案

## 1. 现状与问题

### 1.1 数据流追踪

Pipeline 内部有 word 级时间戳，但导出后丢失：

```
whisper_output.json            ← whisper.cpp 原始输出 ✅
    │
    ▼
transcribe.py → list[Word]     ← 内存中
    │
    ▼
segment.py → SemanticUnit      ← unit.words 保留
    │
    ▼
main.py → SubtitleCue          ← cue.words 保留
    │
    ├── qc.run()               → pipeline 内联 QC 正常 ✅
    │
    └── export → 导出文件       → 不含 words ❌
```

### 1.2 受影响规则

独立 `light-qc` 对导出文件运行时，以下规则**静默跳过**：

| 规则 | 依赖 |
|------|------|
| `EntryPointAccuracy` | `cue.words[0].start` — 入点偏差 |
| `ExitPointPrecision` | `words[-1].end` — 阅读 padding |
| `WordGapAnomaly` | 词间间隔异常 |
| `PaddingConflict` | padding 与下一个词的冲突 |
| `TimeAxisNotOverflow` | 源语 word 边界 |
| `ShotBoundaryHard` | 词在镜头切点分布 |
| `SpeakerConsistency` | word 级多说话人检测 |

### 1.3 命名歧义

当前 `transcript.json` 实际导出的是字幕 cue 列表，与 "transcript" 语义冲突。

### 1.4 耦合风险

如果 light-qc 直接读 `whisper_output.json`，未来替换 ASR 实现（如改用 faster-whisper、SenseVoice 等），light-qc 就必须适配新格式。需要一个 ASR 无关的标准转录格式。

---

## 2. 方案总览

| 动作 | 内容 |
|------|------|
| **重命名** | 现有 `transcript.json` → `cues.json`（cue 列表，不含 words） |
| **新增** | 标准化 `transcript.json`（含 word 时间戳，ASR 无关） |
| **改造** | light-qc `--transcript` 参数读入标准 `transcript.json`，对齐到 cue |

```
pipeline 输出目录：
  asr/
    whisper_output.json     ← whisper.cpp 原始输出（ASR 私有格式）
  cues.json                 ← 字幕 cue 列表（不含 words）
  transcript.json           ← [NEW] 标准化转录（含 word 时间戳）
  segments.json             ← 语义单元（已有，不变）
  en.srt / zh.srt / ...
```

---

## 3. 重命名：现有 transcript.json → cues.json

改动范围小，纯重命名：

- `light-subtitle/main.py`：3 处 `"transcript.json"` → `"cues.json"`
- `README.md`：引用更新

---

## 4. 新增标准化 transcript.json

### 4.1 格式定义

```json
{
  "format": "light-transcript.v1",
  "source": "whisper.cpp large-v3",
  "language": "en",
  "created_at": "2026-05-11T10:30:00Z",
  "words": [
    {"text": "Hello",  "start": 0.00, "end": 0.50, "confidence": 0.98, "speaker": "SPEAKER_00"},
    {"text": "world",  "start": 0.50, "end": 1.00, "confidence": 0.95, "speaker": "SPEAKER_00"},
    {"text": "今天",   "start": 1.20, "end": 1.60, "confidence": 0.92, "speaker": "SPEAKER_01"}
  ],
  "segments": [
    {
      "start": 0.00,
      "end": 1.00,
      "speaker": "SPEAKER_00",
      "text": "Hello world",
      "word_range": [0, 1]
    },
    {
      "start": 1.20,
      "end": 1.60,
      "speaker": "SPEAKER_01",
      "text": "今天",
      "word_range": [2, 2]
    }
  ]
}
```

- `words`：扁平词列表，按 `start` 升序，含 `text / start / end / confidence / speaker`
- `segments`：可选的句子/段落级分组，通过 `word_range [from, to]` 引用 words 索引（便于回溯上下文，对齐引擎不依赖此字段）
- `format` 版本号保证向前兼容

### 4.2 Pipeline 生成位置

在 `light-subtitle/main.py` 中，ASR 完成后、segments 生成后导出：

```python
# main.py 流程中新增:
words = transcribe.run(config, asr_audio)
units = segment.run(words)
export.export_segments(words, units, str(out / "segments.json"))

# [NEW] 导出标准化 transcript
export.export_transcript(words, units, config, str(out / "transcript.json"))
```

`export_transcript()` 在 `light-subtitle/pipeline/export.py` 中新增：

```python
def export_transcript(words, units, config, output_path):
    data = {
        "format": "light-transcript.v1",
        "source": f"whisper.cpp {Path(config.whisper_model).stem}",
        "language": _detect_lang(words),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "words": [
            {"text": w.text, "start": w.start, "end": w.end,
             "confidence": w.confidence, "speaker": w.speaker}
            for w in words
        ],
        "segments": [
            {"start": u.start, "end": u.end, "speaker": u.speaker,
             "text": u.source_text,
             "word_range": [words.index(u.words[0]), words.index(u.words[-1])]
                         if u.words else [0, 0]}
            for u in units
        ],
    }
    export_json_file(data, output_path)
```

### 4.3 为什么不用 whisper_output.json 直接作为 transcript

| 对比维度 | whisper_output.json | transcript.json |
|----------|---------------------|-----------------|
| 格式稳定性 | whisper.cpp 版本升级可能变 | 固定 `light-transcript.v1` |
| ASR 无关 | ✗ whisper 专有字段（tokens/offsets） | ✓ 只有 text/start/end/confidence/speaker |
| 解码负担 | 需要适配多种 token 格式 | 统一 Word 结构 |
| 语言标注 | 无显式 language 字段 | `language` 字段 |
| segments 上下文 | 无 | 可选的 segment 分组 |

---

## 5. light-qc 改造：--transcript 参数

### 5.1 CLI

```bash
--transcript, -t  PATH        标准化 transcript.json 路径
--alignment-tolerance FLOAT   对齐容忍度（秒），默认 0.08（2帧@25fps）
--word-coverage-min FLOAT     最低转录覆盖率阈值，默认 0.95
```

### 5.2 QCConfig 扩展

```python
@dataclass
class QCConfig:
    # ... 现有字段 ...

    transcript_path: str | None = None        # 新增
    alignment_tolerance: float = 0.08          # 新增
    word_coverage_min: float = 0.95            # 新增
```

### 5.3 新增模块：alignment.py

#### `load_transcript(path) -> list[Word]`

```python
def load_transcript(path: str) -> list[Word]:
    """解析 transcript.json 为 Word 列表。"""
    with open(path) as f:
        data = json.load(f)

    fmt = data.get("format", "")
    if not fmt.startswith("light-transcript"):
        raise ValueError(f"Unsupported transcript format: {fmt}")

    return [
        Word(text=w["text"], start=w["start"], end=w["end"],
             confidence=w.get("confidence", 0.0),
             speaker=w.get("speaker"))
        for w in data["words"]
    ]
```

#### `align_words_to_cues(cues, words, tolerance) -> list[Word]`

```
输入: cues: dict[str, list[SubtitleCue]]
      words: list[Word]（按 start 已排序）
      tolerance: float

输出: 未被任何 cue 覆盖的 words

算法:
  1. 对每个 cue:
     a. 二分查找 [cue.start - tolerance, cue.end + tolerance] 内的 words
     b. 记录 word→cue 的覆盖关系
  2. 冲突解决（一个 word 被多个 cue 覆盖）:
     - 按 overlap 占比分配（word 与 cue 的时间重叠 / word 自身时长）
     - 占比相同 → overlap 绝对值更大者
     - 仍相同 → 先出现的 cue
  3. 将最终分配的 words 赋给 cue.words
  4. 返回未被任何 cue 覆盖的 words
```

### 5.4 engine.py 集成

```python
def run_qc(cues, config):
    uncovered_words = []

    # [NEW] Step 0: transcript 对齐
    if config.transcript_path:
        from .alignment import load_transcript, align_words_to_cues
        words = load_transcript(config.transcript_path)
        uncovered_words = align_words_to_cues(cues, words, config.alignment_tolerance)

    # Step 1: 规则引擎（7 条时间对齐规则自动生效）
    issues = RuleEngine(config).check(cues)

    # [NEW] Step 2: 覆盖率检查
    if uncovered_words:
        issues.extend(_build_coverage_issues(uncovered_words, config))
        # 边界对齐由现有规则 EntryPointAccuracy / ExitPointPrecision 覆盖

    # Step 3: [可选] LLM QC ...
```

### 5.5 新增规则

**`TranscriptionCoverage`**（硬性）：
- 对齐后 uncovered_words 非空 → error
- detail: `"转录词 '{text}' @ {time} 未被任何字幕覆盖"`
- fix: `"检查该时间点是否存在漏字幕"`

> 边界对齐（入点/出点）由现有规则 `EntryPointAccuracy` 和 `ExitPointPrecision`
> 覆盖，对齐填充 `cue.words` 后自动生效。

### 5.6 现有规则激活

对齐填充 `cue.words` 后，以下规则无需修改直接生效：

`EntryPointAccuracy` / `ExitPointPrecision` / `WordGapAnomaly` / `PaddingConflict` / `TimeAxisNotOverflow` / `ShotBoundaryHard` / `SpeakerConsistency`

---

## 6. 使用示例

```bash
# 源语言: SRT + transcript
uv run light-qc -i output/en.srt \
  --transcript output/transcript.json

# 翻译验证: 双语 + transcript（验证翻译时间轴）
uv run light-qc -i en.srt -i zh.srt \
  --transcript output/transcript.json \
  --source-lang en --target-lang zh

# 双语模式: 双语 + transcript
uv run light-qc -i en.srt -i zh.srt \
  --transcript output/transcript.json \
  --source-lang en --target-lang zh --bilingual

# 调整参数
uv run light-qc -i en.srt --transcript transcript.json \
  --alignment-tolerance 0.12 --word-coverage-min 0.90 \
  -f html -o report.html
```

---

## 7. 执行流程

```
run_qc(cues, config):

  ┌─────────────────────────────────────────┐
  │ Step 0: Transcript 对齐 [NEW]            │
  │  load_transcript(transcript.json)        │
  │  align_words_to_cues(cues, words)         │
  │  → SubtitleCue.words 被填充               │
  └──────────────┬──────────────────────────┘
                 ▼
  ┌─────────────────────────────────────────┐
  │ Step 1: 规则引擎                          │
  │  RuleEngine(config).check(cues)           │
  │  → 7 条时间对齐规则自动生效                │
  └──────────────┬──────────────────────────┘
                 ▼
  ┌─────────────────────────────────────────┐
  │ Step 2: 新规则 [NEW]                     │
  │  TranscriptionCoverage: 漏字幕检测        │
  │  BoundaryDeviation: 边界偏差              │
  └──────────────┬──────────────────────────┘
                 ▼
  ┌─────────────────────────────────────────┐
  │ Step 3: [可选] LLM QC                    │
  └──────────────┬──────────────────────────┘
                 ▼
  summarize → QCReport
```

---

## 8. 涉及文件清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `light-subtitle/.../main.py` | 改 | `"transcript.json"` → `"cues.json"`（3处）；新增 `export_transcript()` 调用 |
| `light-subtitle/.../pipeline/export.py` | 改 | 新增 `export_transcript()` 函数 |
| `light-qc/.../config.py` | 改 | `QCConfig` 新增 3 个字段 |
| `light-qc/.../alignment.py` | **新增** | `load_transcript()` + `align_words_to_cues()` |
| `light-qc/.../engine.py` | 改 | `run_qc()` 集成对齐 + 新规则 |
| `light-qc/.../rules/sync.py` | 改 | 新增 `TranscriptionCoverage`, `BoundaryDeviation` |
| `light-qc/.../main.py` | 改 | CLI 新增 `--transcript`, `--alignment-tolerance`, `--word-coverage-min` |
| `README.md` | 改 | 重命名 + transcript.json 说明 + `--transcript` 用法 |

---

## 9. Edge Cases

| 场景 | 处理 |
|------|------|
| transcript 与字幕时间轴完全无关 | 覆盖率 < 5%，error: "转录与字幕时间轴差异过大" |
| cue 匹配到 0 个 word | cue.words 保持空，其他规则继续正常执行 |
| 一个 word 被多个 cue 覆盖 | overlap 占比优先分配 |
| 未来 ASR 输出不含 speaker | `speaker` 字段 null，非必填 |
| format 版本不匹配 | 检查 `"light-transcript.vX"` 前缀，未知版本报错 |

---

## 10. 不做的

- **不改 `export_json`（cues 导出）** — cues.json 就是字幕列表，不含 words 是合理的
- **不修改字幕时间轴** — 对齐只填充 `cue.words`
- **不改变 CLI 向后兼容** — `--transcript` 可选
- **不引入新依赖**
- **light-qc 不直接依赖 whisper_output.json** — 通过 transcript.json 解耦
