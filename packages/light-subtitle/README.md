# light-subtitle

字幕能力包 — 语义断句、语言感知分行、样式与字体、字幕/转录导出，以及管线用的 LLM prompt 模板。

## 职责

- **`segment.run(words, max_duration, max_chars_per_line)`** — 六档评分语义断句：词流 → `Segment` 列表（句边界 > 语义 > 停顿 > 从句 > 短语 > 约束溢出；说话人切换必断；Netflix §4 禁断词对）。
- **`language`** — 英语/CJK 断点查找（`EnglishBreakFinder` / `ChineseBreakFinder`）、`split_english` / `split_chinese` 显示分行、语言检测 `detect_source_lang`。
- **`style`** — `SubtitleStyleConfig`（圆角盒主题）、系统字体解析（`resolve_font` / `resolve_font_file`）、盒几何与双语盒式 ASS 事件生成。
- **`cue_builder.build_source_cues(segments, lang)`** — 由 segments 构建源语言 `SubtitleCue`。
- **`export`** — SRT / WebVTT / 单语 ASS / 双语 ASS+VTT / 注释 ASS+VTT / `transcript.json` / `segment.json` / cues JSON 写出。
- **`subtitle.strip_punct`** — 中文显示标点剥离（行末句号移除、逗号→全角空格，保留 `?!…`）。
- **`prompts.render_prompt(name, **kwargs)`** — 渲染包内 `.j2` 模板（plan / translate / evaluate / refine / join / compress / context / annotate）。

## 独立接入示例

```python
from light_subtitle import export, segment
from light_subtitle.cue_builder import build_source_cues
from light_subtitle.language import detect_source_lang
from light_subtitle.language.english import split_english

segments = segment.run(words, max_duration=7.0)        # list[Word] → list[Segment]
lang = detect_source_lang(words)
cues = build_source_cues(segments, lang)               # list[SubtitleCue]

# 显示分行（显式参数，不依赖任何 config 对象）
display_cues = split_english(cues[0], cues[0].text, max_chars_per_line=42, max_lines=2)

export.export_srt(display_cues, "output/video.en.srt")
export.export_transcript(words, segments, "output/transcript.json", source="whisper.cpp")
```

本包不编排流水线；步骤编排、断点续跑与 CLI 在 `light-cli`。
