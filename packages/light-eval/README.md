# light-eval

字幕自改进评估框架（阶段一打样）：针对字幕管线单个步骤（`plan` / `translate`）建立可复跑的评估套件 —— case 发现加载、真实能力包 runner、规则 judge、JSON/HTML 报告。

## 概念

- **Case suite**：`tests/eval/<step>/<case_name>/` 目录约定
  - `case.yaml`：`step` / `kind`（control|edge|boundary）/ `source` / `params`；视频片段 case 另带 `range: {start_unit, end_unit}`
  - `fixture/`：步骤输入 artifact（与管线落盘格式一致；片段 case 只含区间内的 units 及其覆盖的词）
  - `annotation.yaml`（可选）：人工标注（dimensions / defects / overall），供后续 LLM judge 校准
- **Runner**：调 light-subtitle 能力包真实 API 跑单步骤（不 mock），捕获输出 / usage / 耗时 / 异常
- **Judge**：规则 judge（本阶段）按固定维度产出 `DimensionScore`；LLM judge 后续任务接入
- **Report**：JSON（For Agent）+ 单文件 HTML（For Human）

## Fixture 格式

- `plan` 步骤：
  - `fixture/segment.json` —— 管线 `segment/segment.json` 格式（`{"units": [...]}`）
  - `fixture/words.json` —— 全局词级时间轴（word dict 列表，对应 `light_models.word_to_dict`）
- `translate` 步骤：
  - `fixture/plan.json` —— 管线 `plan/plan.json` 格式（plan units）
  - `fixture/glossary.json` / `fixture/summary.json`（可选）

## 使用

```bash
uv run light-eval run tests/eval --step plan -o report.json
uv run light-eval run tests/eval --step translate --llm-api-key $DEEPSEEK_API_KEY -f html -o report.html
```

- `plan`：LLM 可选（`llm=None` 走 deterministic fallback，规则指标仍可评）
- `translate`：无 LLM 时输出空，报告标注 `skipped`

`harvest` / `calibrate` / `serve` 子命令已注册，后续任务实现（harvest：从真实 run 的 output 目录取材生成 case；calibrate：用 annotation.yaml 校准 LLM judge；serve：eval workbench Web 界面）。
