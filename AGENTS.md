# Light — Agent 开发指南

## 项目概览

视频/音频 → 高质量字幕全自动流水线。Monorepo 结构：

```
packages/
├── light-models/        共享数据契约（Word, Segment, SubtitleCue, is_cjk…）
├── light-subtitle/      ASR → 翻译 → 字幕流水线
│   ├── pipeline/        ASR → correct → punct → segment → translate → subtitle → export
│   ├── step_registry.py / step_plan.py / run_state.py / state_hydrate.py  # 步骤注册与 resume
│   └── language/        语言处理（英语/CJK 断句、标点、显示约定）
├── light-qc/            独立 QC 引擎（规则 + LLM）
├── light-regression/    回归测试工具（固定黄金基线 + rebaseline）
├── light-backend/       FastAPI Web 后端（routers/ + services/）
└── light-frontend/      React + Vite SPA（pages/ + components/）
```

## 按改动路由（动手前先看）

改了哪个包，就**必须**跑对应的验证。命令直达，无占位符：

| 改动模块 | 必跑验证 | 命令 |
|---|---|---|
| light-subtitle | 回归测试（已内置 QC） | `uv run light-regression run tests/regression/cases/<case>/case.yaml` |
| light-qc | 端到端 QC | `uv run light-qc -i <本地 output 里的 .srt> --transcript <本地 output 里的 transcript.json> -f json` |
| light-frontend | 类型检查 + 构建 | `npm --prefix packages/light-frontend run build` |
| light-backend | Lint + 后端测试 | `uv run ruff check packages/light-backend/ && uv run pytest tests/test_light_backend_playback.py -v` |
| 任何 Python 改动 | Lint + Format + 单测 | `uv run ruff check . && uv run ruff format --check . && uv run pytest tests/ -v` |

> **重要**：上表是"按改动类型"叠加的，不是二选一。例如改 light-subtitle，既要跑"任何 Python 改动"那行，也要跑"light-subtitle"那行。验收前**全部**必须通过，缺一项不算完成。

## 回归测试

### Case 速查

`tests/regression/cases/` 下共 7 个 case，分 smoke（快速冒烟，首选）与完整（长音频，大改动用）：

| Case | 时长 | 场景 | 定位 |
|---|---|---|---|
| `yt_first5min` | ~5min | en 单语 | **smoke 首选**（改 ASR/断句/标点/导出） |
| `yt_first5min_scene_b` | ~5min | en→zh 翻译 | **smoke 首选**（改翻译/字幕格式） |
| `ilya_test_5min` | ~5min | en 单语 | smoke（不同音源验证 ASR） |
| `ilya_test_5min_scene_b` | ~5min | en→zh 翻译 | smoke（不同音源验证翻译） |
| `yt_kYkIdXwW2AE` | ~37min | en 单语 | 完整验证 |
| `yt_kYkIdXwW2AE_scene_b` | ~37min | en→zh 翻译 | 完整验证 |
| `ilya_sutskever_age_of_research_scene_b` | ~30min | en→zh 翻译 | 完整验证（最长） |

**默认选 smoke case**；只有改动可能影响整条管线、或 smoke 无法覆盖的场景，才跑完整 case。

### 黄金基线机制

回归测试采用**固定黄金基线**比对（非滚动比对）：

- 每个 case 在 `tests/regression/snapshots/<case>/baseline.json` 存一份黄金基线报告
- `light-regression run` 每次跑都跟这份固定基线比，跨人/跨机器可比
- 首跑（无基线）直接 PASS；之后任何回归都会被捕获
- 代码改进、确认输出质量达标后，用 `rebaseline` 推进基线：
  ```bash
  uv run light-regression rebaseline <case.yaml>                # 重跑一次并设为新基线
  uv run light-regression rebaseline <case.yaml> --from-run <run_id>  # 用已有 run 设基线（不重跑）
  ```

### 复现基线的环境要求

黄金基线的可复现性依赖**固定的 ASR/LLM 环境**。复现 baseline 需要：

- **ASR**：whisperx（默认 `--asr-engine`）或 whisper.cpp，模型 `ggml-large-v3-turbo.bin`（CLI 默认 `--whisper-model`）
- **LLM**：默认模型 `deepseek-v4-flash`（`--llm-model`），需配置 `--llm-base-url` 与 `--llm-api-key`（或对应环境变量）
- **Diarization**：pyannote（`--diarize-model`）

> 实测同环境重跑计数（Errors/Warnings/Suggestions）完全一致，PASS/FAIL 判定确定；逐条 detail 可能有亚毫秒浮点抖动，不影响判定（`checker.py` 仅按计数与规则分组数量判定 `degraded`）。

换 ASR 模型/LLM 提供商后，基线计数可能漂移，需 `rebaseline` 重建。

## 技术栈与规范

### Python 模块（light-models / subtitle / qc / regression / backend）

**技术栈**：Python 3.12+, uv workspace, hatchling 构建

**代码规范**：
- 使用 `from __future__ import annotations`
- 类型注解必需
- 模块/函数级 docstring + 关键逻辑注释
- Import 分组：stdlib → third-party → local（空行分隔）
- 字符串用双引号
- 函数/类之间用 `──` 风格分隔注释
- line-length: 120（ruff 强制执行）

**Lint / Format**：

| 命令 | 说明 |
|---|---|
| `uv run ruff check .` | 检查代码 |
| `uv run ruff check --fix .` | 自动修复问题 |
| `uv run ruff format .` | 格式化代码 |
| `uv run ruff format --check .` | 检查格式（CI 用） |

### 前端模块（light-frontend）

**技术栈**：React 19, Vite 6, TailwindCSS 4, TypeScript 5.7 strict mode

**代码规范**：
- 函数组件 + hooks，避免 class 组件
- JSX 属性使用双引号
- Tailwind 原子类优先，避免自定义 CSS
- 使用 SWR 管理 API 请求状态
- TypeScript strict 模式，`tsc -b` 通过方可提交

**Lint / 类型检查**：

| 命令 | 说明 |
|---|---|
| `tsc -b` | TypeScript 类型检查（含在 `npm run build` 中） |
| `npm run build` | 类型检查 + Vite 构建 |

## 提交规范

- `feat:` 新功能
- `fix:` 修复
- `test:` 测试相关
- `refactor:` 重构
- `docs:` 文档

## CLI 使用

见 `--help`：

```bash
uv run light-subtitle --help
uv run light-qc --help
uv run light-regression --help
```

## 断点续跑

CLI 支持 `--resume`（读 `pipeline_run.json`）和 `--resume-from STEP`（从指定步骤开始，此前步骤 hydrate 不执行）。Web 后端暂未接入。

改 resume 逻辑：`step_registry.py`（step 定义）→ `step_plan.py`（plan/校验）→ `state_hydrate.py`（灌状态）。验证：`uv run pytest tests/test_run_state.py -v`。

用户向说明见 README；开发迭代示例：

```bash
uv run light-subtitle -i <input> -o output --resume-from segment          # 跳过 ASR，需 transcript.json
uv run light-subtitle -i <input> -o output --target-lang zh --resume-from subtitle  # 跳过 ASR+翻译，需 raw.json
```

勿删 `transcript.json`、`translations/raw.json` 等 resume 依赖 artifact。

## 工作流

每个工作流已内联该模块的必跑验证。任何 Python 改动另加通用检查：`uv run ruff check . && uv run ruff format --check . && uv run pytest tests/ -v`。

### light-subtitle 开发工作流

**目标**：提高字幕管线输出质量

1. 接受具体需求
2. 制定计划（向用户确认）
3. 跑回归基线：`uv run light-regression run tests/regression/cases/<smoke case>/case.yaml`（改动落在哪段就选对应 smoke，见「回归测试·Case 速查」）
4. 进行开发
5. 单元测试：`uv run pytest tests/ -v`
6. **回归测试**（与基线对比，命令同第 3 步；小改动用 `--resume-from` 跳过 ASR，见「断点续跑」）。`light-regression run` 内部已子进程调 `light-qc`，**无需再单独跑 QC**
7. 与基线对比，验证结果（不符合预期 → 返回步骤 4）
8. 确认代码改进、输出质量达标且需推进基线时，`uv run light-regression rebaseline <case.yaml>`
9. 完成功能，报告结果，向用户确认是否提交

首次全量跑生成 artifact；迭代翻译/格式化时用 `--resume-from`（见上文「断点续跑」）。

### light-qc 开发工作流

**目标**：提高 QC 管线质量

1. 接受具体需求
2. 制定计划（向用户确认）
3. 跑当前端到端基线：`uv run light-qc -i <本地 output 里的 .srt> --transcript <本地 output 里的 transcript.json>`（QC 独立端到端检查仅改 light-qc 时需要；改 light-subtitle 由回归内置覆盖）
4. 进行开发
5. 单元测试：`uv run pytest tests/test_rules.py -v`
6. **端到端 QC**（命令同第 3 步，输入同基线）
7. 与基线对比，验证结果（不符合预期 → 返回步骤 4）
8. 完成功能，报告结果，向用户确认是否提交

> light-qc 建议始终携带 `--transcript` 参数，以启用完整的时间轴对齐规则。

### light-backend / light-frontend 开发工作流

**目标**：Web 界面功能开发与调试

1. 接受具体需求
2. 制定计划（向用户确认）
3. 启动后端：`uv run light-backend`
4. 启动前端 dev server：`npm --prefix packages/light-frontend run dev`
5. 进行开发
6. Lint：`uv run ruff check packages/light-backend/` / 前端自动热更新
7. **前端构建**：`npm --prefix packages/light-frontend run build`（类型检查 + Vite 构建，必须通过）
8. 端到端验证：前端提交 URL → 管线运行 → 播放
9. 完成功能，报告结果，向用户确认是否提交

> **开发提示**：
> - 前端 Vite 代理自动将 `/api` 转发到后端 `localhost:8787`
> - 后端绑定 `0.0.0.0`，手机可通过局域网 IP 访问测试
> - 修改 `light-subtitle` 步骤注册或 orchestrator 后需重启后端
> - 导入已有 output 目录用于快速验证（跳过下载和管线）

## 查看报告

| 报告类型 | 生成命令 | 打开方式 |
|---|---|---|
| QC 报告 (HTML) | `uv run light-qc -i <文件> --transcript <transcript> -f html -o output/qc_report.html` | `open output/qc_report.html` |
| 回归 Dashboard | `uv run light-regression dashboard` | `open regression_dashboard.html` |

## 关键约束

- `output/` 已 gitignore，用于本地验证和测试输出
- `data/` 已 gitignore，用于 Web 后端运行时数据（SQLite + 视频文件）
- 回归测试快照 `tests/regression/snapshots/` **进 git 共享**，是固定黄金基线，**禁止删除**；质量改进后用 `rebaseline` 推进，不要手动删快照
- 新 QC 规则必须零误报才提交
- light-qc 独立端到端检查仅改 light-qc 时需要；改 light-subtitle 由回归内置覆盖
- resume 见上文「断点续跑」；改 step 注册/hydrate 后跑 `tests/test_run_state.py`
- 修改涉及 CLI 参数、行为、输出格式等外部可见变更时，须同步更新 `README.md` 对应章节
- **不要**擅自提交代码，除非用户要求提交再提交
