---
status: 待实施
---

# 判定模型缺陷化（defect gating）

字幕评估的通过判定，由「是否存在确认的必改缺陷」决定，而非由 1-5 平均分决定。

## 现状与问题

- `DimensionScore.passed = score >= 4`（`judges/llm.py`），case 通过 = 全维度通过（`models.py`）。1-5 分是维度上的平均感受：一个必改缺陷（如从句被斩头）拉不低 4 分，case 被放过。
- 判定逻辑有三套且互不相同：`llm.py PASS_THRESHOLD=4`、`serve.py` suggested_overall 硬编码 4/3、工作台人工 overall 纯手填（与缺陷卡片 verdict 无派生关系）。
- judge 的 issues 列表（真正的缺陷信号）不参与 gate；校准（calibration.py）也只看 score 不看 issues。
- 关联历史：judge issues 已经过一轮降噪（unit_id 归属校验、merge 可行性复算、score=5 清空 issues、speaker/summary 上下文补齐、中位数聚合、自然序排序），精度已具备承担 gate 的条件。

## 目标

- gate 缺陷化：任一 must_fix 缺陷 → fail；score 降级为信息与校准输入（calibration MAE 指标不变）。
- 判定单一事实源：eval run、工作台 suggested_overall、人工派生预填共用同一判定函数。
- 缺陷分级：`must_fix`（影响观看/理解，必须改）与 `minor`（有瑕疵但不影响观看）。

## 契约变更

### judge issue schema

```json
{"unit_id": "p0014_2", "problem": "…", "severity": "must_fix"}
```

- `severity` 必填，枚举 `must_fix | minor`；缺失/非法值按 `must_fix` 处理（宁严勿宽）。
- prompt（judge_plan.j2 / judge_translate.j2）给出分级定义与示例：
  - must_fix：语义断裂/斩头、漏译增译、说话人混排、术语错误——观众会注意到
  - minor：风格层面可更优但不影响理解
- 现有后过滤层全部保留（unit_id 归属、merge 可行性复算、score=5 清空 issues）。

### 通过判定

| 层 | 规则 |
|---|---|
| 规则指标（rules.py） | 不变（本来就是缺陷计数 gate） |
| LLM judge 维度 | `passed = 无 must_fix issue` |
| case | 全维度 passed（不变） |
| suggested_overall（serve） | any must_fix → fail；仅 minor → borderline；无 issue → pass（删除硬编码 4/3，与 judge 共享判定函数） |

score 保留产出：校准 MAE/±1 一致率、报告展示、工作台维度分预选，均不变。

### 人工侧（工作台 + annotation）

- 缺陷卡片带 severity 标签（AI 预填）；`annotation.yaml` 的 defects 增加可选 `severity` 字段（旧文件无此字段按 minor 读，向后兼容）。
- 保存时 overall 仍取 select 值，但预填由「卡片 verdict + severity」派生：任一「通过」的 must_fix 卡 → fail；任一「通过」卡 → borderline；否则 pass。人工可覆盖。
- `has_judge_suggestion` 展示逻辑不变。

### 文档同步

- `success_criteria.md`：判定节改写为缺陷化 gate + severity 定义；顺带修正 target_lang_ratio 阈值 0.8→0.6（与代码一致）。
- `annotation_guide.md`：补 severity 分级与 overall 派生规则。

## 影响面

`judges/llm.py`、`prompts/judge_plan.j2`、`prompts/judge_translate.j2`、`serve.py`、`web/index.html`、`models.py`（Annotation severity 兼容）、`report.py`（维度行展示 must_fix 计数）、上述两份 docs。

## 测试

- judge：severity 解析与默认值（缺失按 must_fix）、must_fix 一票否决、minor 放行、后过滤与 severity 的交互
- serve：suggested_overall 三态派生、annotation severity 读写兼容
- models：Annotation 含/不含 severity 的 roundtrip
- 既有：rules/校准相关用例应全部保持绿色

## 范围外

- 缺陷级校准（judge issues vs 人工 defects 的 precision/recall）：另立迭代
- score 的彻底移除：待缺陷级校准上线后再评估
