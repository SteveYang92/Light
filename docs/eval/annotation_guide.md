# 黄金集标注规范

在人工工作台（`light-eval serve`）完成选样与标注。标注结果写入每个 case 的 `annotation.yaml`，是校准 LLM judge 的基准。

## 选样（候选页 → 选段）

目标 10–20 个 case，两个步骤（plan / translate）。**case = 视频的连续片段 + kind**：在候选行点「选段」，加载 unit 列表后点两个端点选定区间，选 kind 建 case。同一个视频可以切多段——干净段落做 control，问题密集段落做 edge，一段视频贡献多个 case。

- **control**（约一半）：内容干净、管线历史表现好的片段
- **edge**：QC 报警多的、长难句多的、术语密集的、语速快的片段
- **boundary**（1–2 个即可）：特殊场景（大量代码/公式朗读、多语言混杂、极短词碎片）

片段建议 20–60 个 unit：太小缺乏上下文，太大评审负担重。

## 标注（Cases 详情页）

先点 **Run** 生成步骤输出，再逐项标注。

### 维度分（1-5，锚点见 success_criteria.md）

- plan：`boundary_quality`、`split_necessity`
- translate：`faithfulness`、`naturalness`、`unit_integrity`、`terminology`

打分原则：
- 按**观众体验**打分，不按"有没有触发某规则"
- 3 分是关键分水岭：明显注意到问题但不妨碍理解
- 拿不准就 low 不 high——judge 校准时宁可见严

### 缺陷记录（defects）

对具体有问题的 unit：选 unit_id + 一句话问题描述 + severity 分级。例：

```yaml
defects:
  - unit_id: p0042
    issue: 在 "which means" 前断开，从句被截成两半
    severity: must_fix   # 必改：影响观看/理解（语义断裂/斩头、漏译增译、说话人混排、术语错误）
  - unit_id: p0107
    issue: "transformer" 译成"变形金刚"，应为"变换器"
    severity: minor      # 次要：风格层面可更优但不影响理解
```

- `must_fix` 与 `minor` 的定义与 judge 输出契约一致（见 success_criteria.md「缺陷分级」）；AI 预评会预填 severity，人工按观众体验复核
- 缺陷记录用于：judge 证据能力校准 + 阶段二失败模式分类。发现多少记多少，不需要穷尽

### 总体判定（overall）

- `pass`：可以直接发布的质量
- `borderline`：有瑕疵但能看
- `fail`：不可接受

派生规则（工作台在判定缺陷卡片时自动预填，人工可覆盖）：任一收录的 must_fix 缺陷 → `fail`；任一收录缺陷 → `borderline`；无收录缺陷 → `pass`。

## 标注后的校准

```bash
uv run light-eval calibrate tests/eval -o tests/eval/calibration_report.json
```

±1 分一致率 ≥ 80% 认定 judge 可信；不达标则迭代 rubric（`packages/light-eval/src/light_eval/prompts/judge_*.j2`）后重跑校准。
