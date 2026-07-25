# light-asr-polish

LLM-based ASR post-processing — transcript correction and punctuation restoration over word-level ASR output.

## 职责

- **`correct(words, llm, work_dir=None, progress=None)`** — 转录纠正：先对全文提取领域上下文（domain / topics / terminology），再按停顿分段、批处理调用 LLM 修同音字、专有名词、重复词与语法词形，时间戳保持不变（词数变化时按比例重分配）。
- **`restore_punct(words, llm, work_dir=None, progress=None)`** — 标点还原：对齐后的 whisper 词级输出常缺标点，按停顿分段批量送 LLM 补标点，再经字符级 diff 映射回词。
- **`word_segments`** — 两阶段共用的按停顿分段工具（`WordSegment`、`group_words_by_gap`、`join_word_text`、`merge_short_segments`）。

两个 API 都返回 `(words, usage | None)`；`work_dir` 给出时写调试 artifact（`transcript_correct/`、`punct_restore/`），为 `None` 时完全跳过落盘。

## 独立接入示例

```python
from light_asr import transcribe
from light_asr_polish import correct, restore_punct
from light_llm.client import OpenAIClient

words = transcribe("input.mp4", ...)          # list[Word]，来自 light-asr
llm = OpenAIClient(base_url=..., api_key=..., model=...)

words, correct_usage = correct(words, llm, work_dir="output")
words, punct_usage = restore_punct(words, llm, work_dir="output")
```

是否启用（LLM key 是否存在、开关是否打开）由调用方门控；不满足时不要调用本包。
