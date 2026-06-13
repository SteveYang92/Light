# light-qc 规则说明

> 参考规范：`docs/subtitle.md` §1.2-§1.3, §7.3

---

## 规则分类

| 分类 | 说明 |
|------|------|
| **硬性规则** | 不可妥协，违反即为 `error`，报告 `FAILED` |
| **柔性策略** | 建议改进，`warning` 需关注、`suggestion` 可酌情忽略 |

---

## 一、硬性规则

### 1.1 字幕格式与时长

#### MaxLines

- **检测**：字幕超过最大行数限制（中文/英文默认 2 行）
- **场景**：全部
- **规范**：`subtitle.md §1.2` — 最大 2 行
- **阈值**：`--max-lines`（默认 2）

#### MinDuration / MaxDuration

- **检测**：字幕时长超出允许范围
- **场景**：全部
- **阈值**：`--min-duration`（默认 0.8s）、`--max-duration`（默认 7.0s）
- **原理**：过短的字幕来不及阅读；过长的字幕占据屏幕过久

#### Overlap

- **检测**：相邻两条字幕时间重叠（后条 start < 前条 end）
- **场景**：全部
- **原理**：同时显示两条字幕会造成阅读混乱

#### EmptyText

- **检测**：字幕文本为空或仅为空白字符
- **场景**：全部

---

### 1.2 行长与阅读速度

#### ChineseLineLength / EnglishLineLength

- **检测**：中/英文单行超出字符限制
- **场景**：全部
- **阈值**：`--max-chars-zh`（默认 16 汉字）、`--max-chars-en`（默认 42 字符）
- **规范**：`subtitle.md §1.2`
- **说明**：只统计 CJK 字符（中文），与总字符数（英文）分开计算

#### ReadingSpeed

- **检测**：阅读速度（CPS = 字符数/秒）超出上限
- **场景**：全部
- **阈值**：`--cps-limit`（默认 9 字/秒 中文）、`--cps-limit-en`（默认 25 字符/秒 英文）
- **规范**：`subtitle.md §1.2`
- **说明**：中文只统计 CJK 字符；英文统计全部非换行字符

#### MissingPunctuation

- **检测**：中文句尾缺少标点（。？！…）
- **场景**：全部（仅中文 cue）
- **豁免**：如果下一条字幕紧随其后（间隔 ≤ 0.8s），视为句意未完整，不报错

---

### 1.3 时间轴同步

所有时间轴规则 **依赖 `cue.words`（词级时间戳）**，需通过 `--transcript` 参数加载 `transcript.json` 激活。

#### EntryPointAccuracy

- **检测**：字幕入点（`cue.start`）与首个词的起始时间偏差
- **场景**：全部（需 `--transcript`）
- **阈值**：`entry_tolerance_frames / fps`（默认 2 帧 @ 25fps = 0.08s）
- **规范**：`subtitle.md §1.3-1` — 入点尽量贴近说话开始
- **算法**：`|cue.start - words[0].start| > tolerance`

#### TimelineGap

- **检测**：同说话人的相邻 cue 之间存在微小间隙（< `min_gap`），应做 chaining
- **场景**：全部（需 `--transcript`）
- **阈值**：`--min-gap`（默认 0.1s）
- **原理**：同说话人连续两句之间的 < 0.1s 间隙会导致闪烁，应连接为一条

#### GapFlash

- **检测**：相邻字幕之间的间隔过短，可能导致闪烁
- **场景**：全部
- **阈值**：`--min-gap`（默认 0.1s）
- **与 TimelineGap 的区别**：`GapFlash` 对所有相邻 cue 生效（不管说话人），`TimelineGap` 只对同说话人的情况建议 chaining

#### ShotBoundaryHard

- **检测**：字幕跨越镜头切点，且台词在该处有自然断点
- **场景**：全部（需 `config.shot_changes` + `--transcript`）
- **规范**：`subtitle.md §1.3-4` — 字幕尽量不跨镜头；台词没有跨镜头时，字幕也尽量在镜头切换前结束
- **算法**：检查 cue 内的 words 在切点两侧的分布；如果两侧都有词且间隙 < 0.3s 则判定为自然桥接，不报错

---

### 1.4 翻译完整性（Scene B / C）

#### TranslationCompleteness

- **检测**：每个源语言语义单元都有对应的翻译
- **场景**：B / C
- **两种模式**：
  - **unit_id 模式**（pipeline）：按 `unit_id` 做集合差集，精确定位缺失单元
  - **时间 overlap 模式**（独立 SRT/VTT）：按时间重叠检测有无翻译覆盖
- **同时检测**：孤立的翻译单元（有翻译但无对应源语），可能是 LLM 幻觉

#### TimeAxisNotOverflow

- **检测**：翻译字幕的时间轴不超出源语言 word 时间范围
- **场景**：B
- **原理**：翻译是在源语基础上叠加的，不能早于源语开始或晚于源语结束

---

### 1.5 双语检查（Scene C）

#### BilingualMapping

- **检测**：每条 cue 在另一语言中至少有一个时间重叠的对应 cue
- **场景**：C
- **原理**：双语字幕应一一对应，不允许孤立的单语 cue

#### CombinedReadingSpeed

- **检测**：中英文双语同时显示时，每种语言的 CPS 独立不超限
- **场景**：C
- **阈值**：`--cps-limit` + `--cps-limit-en`
- **原理**：两条语言共享同一显示窗口，各自仍需满足阅读速度

#### VisualDensity

- **检测**：双语总视觉密度（中文字符 + 英文字符 / overlap 时长）≤ 18 字符/秒
- **场景**：C
- **原理**：两条字幕同时显示时，总信息量不宜过大

#### LineBalance

- **检测**：中英文行数差距 ≤ 2
- **场景**：C
- **原理**：行数差距过大会导致视觉不协调

---

### 1.6 转录覆盖率

#### TranscriptionCoverage

- **检测**：原始转录中有词未被任何字幕 cue 覆盖
- **场景**：全部（需 `--transcript`）
- **阈值**：`--word-coverage-min`（默认 0.95）
- **分级报告**：
  - 全局覆盖率 < 阈值 → `error`，提示可能存在大段漏字幕
  - 逐个未覆盖词 → `warning`（最多 30 个，超出省略）
- **原理**：如果原始转录中的词没有出现在任何字幕中，说明该时间点缺失字幕

---

## 二、柔性规则

### 2.1 语义与换行

#### SemanticBreaks

- **检测**：双行字幕首行过短（中文 ≤ 3 字 / 英文 ≤ 5 字符），语义切分不自然
- **场景**：全部
- **原理**：断行应在语义分界处，避免一个词孤悬在另一行

#### OrphanWords

- **检测**：英文第二行仅一个词（orphan word）；中文第二行 ≤ 2 字
- **场景**：全部
- **原理**：孤词影响阅读节奏，应调整断行位置

#### CompoundWords

- **检测**：复合词/术语被换行拆散
- **场景**：全部
- **术语来源**：内置列表 + `--glossary` YAML 中的术语
- **内置复合词示例**：人工智能、machine learning、deep learning 等

---

### 2.2 时间轴精度（需 --transcript）

#### ExitPointPrecision

- **检测**：字幕出点与末词的阅读 padding 是否合理
- **场景**：全部（需 `--transcript`）
- **阈值**：
  - padding < 0.15s → `warning`（出点紧贴语音结束，无阅读时间）
  - padding > 1.0s  → `suggestion`（冗余静默过长）
- **规范**：`subtitle.md §1.3-2` — 出点可适当延后
- **算法**：`padding = cue.end - max(words.end)`

#### WordGapAnomaly

- **检测**：cue 内部出现异常大的词间间隔
- **场景**：全部（需 `--transcript`）
- **算法**：
  1. 计算所有词间间隔（`word[i+1].start - word[i].end`）
  2. 排除零间隔（whisper 零时长词 artifact）
  3. 取中位数作为"典型间隔"
  4. 任何 > 5× 中位数 且 > 0.20s 的间隔 → `suggestion`
- **原理**：异常大间隔可能是漏词、ASR 对齐错误，或适合切分字幕的自然断句点
- **调参**：`MIN_GAP_ABSOLUTE` = 0.20s，乘数 = 5×

#### PaddingConflict

- **检测**：当前 cue 的阅读 padding 侵入了下一个 cue 的语音起始
- **场景**：全部（需 `--transcript`）
- **算法**：
  1. 计算当前 cue 末词到下条 cue 首词的"说话间隙"
  2. 如果说话间隙 < 0.3s 且当前 padding > 说话间隙 → `suggestion`
- **规范**：`subtitle.md §3.2` — 如果下一条字幕很近，则压缩 padding

---

### 2.3 说话人与翻译

#### SpeakerConsistency

- **检测**：单条字幕包含多个说话人
- **场景**：全部（需 `--transcript`，依赖 word 级 speaker 标注）
- **原理**：一条字幕应只有一个说话人；多人对话应切分为多条

#### ShotChangeSoft

- **检测**：字幕跨镜头切点
- **场景**：全部（需 `config.shot_changes`）
- **与 ShotBoundaryHard 的区别**：`Soft` 放宽了判断条件，对所有跨镜头的 cue 发出 `suggestion`

#### TranslationQuality

- **检测**：机翻风格表达、填充词、过度压缩
- **场景**：B / C
- **检测项**：
  - 机翻连接词：moreover / nevertheless / furthermore / consequently 等
  - 填充词：well / you know / i mean / 就是说 / 那个 / 嗯 / 呃
  - 单汉字 cue：可能压缩过度丢失信息

#### TerminologyConsistency

- **检测**：`--glossary` 中的术语在源语/译语中是否一致
- **场景**：B / C
- **原理**：源语术语不应出现在译语中，反之亦然

#### BilingualBalance

- **检测**：中英文断行结构是否一致（避免逐行硬对齐）
- **场景**：C
- **原理**：中英文的断行应各自遵循语言习惯，而非逐行机械对齐

---

## 三、规则注册方式

规则在 `rules/registry.py` 的 `RuleEngine._register()` 中集中注册，按场景选择性激活：

```python
# 所有场景
self._hard_rules = [
    MaxLines, ChineseLineLength, ReadingSpeed,
    MinDuration, MaxDuration, Overlap, EmptyText, MissingPunctuation
]

# Scene A / B / C 都激活
self._hard_rules.append(GapFlash)

# 时间轴同步规则（所有场景，需 --transcript）
self._hard_rules.append(EntryPointAccuracy)
self._hard_rules.append(TimelineGap)

# Scene B 额外规则
self._hard_rules.append(TimeAxisNotOverflow)
self._hard_rules.append(TranslationCompleteness)

# Scene C 额外规则
self._hard_rules.append(BilingualMapping)
self._hard_rules.append(CombinedReadingSpeed)
# ...
```

---

## 四、LLM QC（可选）

启用 `--llm` 后，额外调用 LLM 对字幕进行质量检查。LLM 发现的新问题会以 `[LLM]` 后缀标注，与规则引擎的结果去重（规则引擎优先）。

- **分块**：每 50 条 cue 一个 batch
- **去重**：`(rule, cue_id)` 联合键，规则引擎已报告的问题不再重复

---

## 五、相关文档

| 文档 | 说明 |
|------|------|
| `docs/subtitle.md` | 字幕制作规范（规范依据） |
| `docs/light_qc_transcript_alignment.md` | `--transcript` 对齐方案设计 |
| `README.md` | 使用指南与 CLI 参数 |
