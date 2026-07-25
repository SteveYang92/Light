# 关键步骤成功标准（eval 考试大纲）

本文件定义字幕管线关键步骤的输入→输出契约、质量维度与 pass/fail 阈值。
评估实现见 `packages/light-eval`，case 集见 `tests/eval/`。

评分分两层：
- **规则指标**（确定性，硬门槛）：计算值精确可复现，违反即 fail
- **LLM judge 维度分**（软门槛）：1-5 分，≥4 为 passed

## plan（断句规划）

输入：pause segments（`list[Segment]`）→ 输出：cue units（`list[Segment]`，unit_id `pNNNN` / `pNNNN_K`）

### 规则指标（硬门槛）

| 维度 | 判定 | passed |
|---|---|---|
| `word_coverage` | 输入全部词被 unit 覆盖 | 覆盖率 = 100% |
| `duration_violations` | unit 时长超出 [min_duration, max_duration] 的数量 | = 0 |
| `dangling_tails` | unit 以功能词结尾（dangling tail）的数量 | = 0 |
| `empty_units` | 空文本 unit 数 | = 0 |

### LLM judge（软门槛，≥4 分）

> plan 步骤自身有时长硬约束（min 0.8s / max 7.0s、约 48 字符两行上限）。judge 评的是**约束可行域内的语义最优**，不是无约束理想标准——被时长预算逼出来的从句间断点是合理选择，不扣分；时长违规由规则指标判定，judge 不重复扣。

| 维度 | 含义 |
|---|---|
| `boundary_quality` | 在时长预算可行域内选择了语义最优边界；仅当存在满足预算且语义更好的边界却未用时扣分 |
| `split_necessity` | 拆分/合并必要性：不过碎（可合法合并的碎 unit）不过长（有合规拆分点却硬塞多句）；时长违规本身不归此维度 |

## translate（翻译）

输入：cue units + glossary/content_summary → 输出：译文 cues（`list[SubtitleCue]`）

### 规则指标（硬门槛）

| 维度 | 判定 | passed |
|---|---|---|
| `unit_coverage` | 输出覆盖输入 unit 的比例 | = 100% |
| `empty_translations` | 空文本 cue 数 | = 0 |
| `target_lang_ratio` | 目标语言字符占比（zh 用 CJK 判定） | ≥ 阈值（默认 0.8） |
| `source_fidelity` | cue 的 unit_id 链/顺序/时间窗与输入一致 | 一致 |

### LLM judge（软门槛，≥4 分）

| 维度 | 含义 |
|---|---|
| `faithfulness` | 忠实度：不漏译、不增译、不歪曲原意 |
| `naturalness` | 口语自然度：像母语者说的话，无翻译腔 |
| `unit_integrity` | 单元完整性：不串单元、unit_id 对应正确 |
| `terminology` | 术语一致性：遵循 glossary，前后一致 |

## judge 评分锚点（两步骤统一）

| 分 | 含义 |
|---|---|
| 5 | 无问题 |
| 4 | 轻微问题，不影响观看 |
| 3 | 明显问题，但可理解 |
| 2 | 影响理解 |
| 1 | 严重错误 |

## case 分级（playbook 三类）

- `control`：模型擅长的干净样本，应永远通过——回归检测器
- `edge`：曾失败的场景（QC 报警多、长难句、密集术语），确保历史错误不重演
- `boundary`：应触发兜底/保守行为的场景，验证系统知道何时不乱来
