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
├── light-regression/    回归测试工具
├── light-backend/       FastAPI Web 后端（routers/ + services/）
└── light-frontend/      React + Vite SPA（pages/ + components/）
```

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

### light-qc 开发工作流

**目标**：提高 QC 管线质量

1. 接受具体需求
2. 制定计划（向用户确认）
3. 跑当前端到端基线：`uv run light-qc -i <输入> --transcript <transcript.json>`
4. 进行开发
5. 单元测试：`uv run pytest tests/test_rules.py -v`
6. 端到端测试（输入同基线）
7. 与基线对比，验证结果（不符合预期 → 返回步骤 4）
8. 完成功能，报告结果，向用户确认是否提交

### light-subtitle 开发工作流

**目标**：提高字幕管线输出质量

1. 接受具体需求
2. 制定计划（向用户确认）
3. 跑回归测试基线：`uv run light-regression run <用户指定 case.yaml>`
4. 进行开发
5. 单元测试
6. 回归测试（输入同基线）
7. 与基线对比，验证结果（不符合预期 → 返回步骤 4）
8. 完成功能，报告结果，向用户确认是否提交

首次全量跑生成 artifact；迭代翻译/格式化时用 `--resume-from`（见上文「断点续跑」）。

### light-backend / light-frontend 开发工作流

**目标**：Web 界面功能开发与调试

1. 接受具体需求
2. 制定计划（向用户确认）
3. 启动后端：`uv run light-backend`
4. 启动前端 dev server：`npm --prefix packages/light-frontend run dev`
5. 进行开发
6. Lint：`uv run ruff check packages/light-backend/` / 前端自动热更新
7. 端到端验证：前端提交 URL → 管线运行 → 播放
8. 完成功能，报告结果，向用户确认是否提交

> **开发提示**：
> - 前端 Vite 代理自动将 `/api` 转发到后端 `localhost:8787`
> - 后端绑定 `0.0.0.0`，手机可通过局域网 IP 访问测试
> - 修改 `light-subtitle` 步骤注册或 orchestrator 后需重启后端
> - 导入已有 output 目录用于快速验证（跳过下载和管线）

## 代码修改后必做

每次修改代码后，以下检查**必须全部通过**才能报告结果。若修改涉及 CLI 参数、行为、输出格式等外部可见变更，须同步更新 `README.md` 对应章节。

### Python 通用

| 检查项 | 命令 |
|---|---|
| Lint | `uv run ruff check .` |
| Format | `uv run ruff format --check .` |
| 单元测试 | `uv run pytest tests/ -v` |

### 前端

| 检查项 | 命令 |
|---|---|
| 类型检查+构建 | `npm --prefix packages/light-frontend run build` |

### 模块专项

| 检查项 | 命令 | 适用场景 |
|---|---|---|
| 端到端 QC | `uv run light-qc -i <输入> --transcript <transcript.json>` | light-qc 修改 |
| 回归测试 | `uv run light-regression run <case.yaml>` | light-subtitle 修改 |

## 查看报告

| 报告类型 | 生成命令 | 打开方式 |
|---|---|---|
| QC 报告 (HTML) | `uv run light-qc -i <文件> --transcript <transcript> -f html -o output/qc_report.html` | `open output/qc_report.html` |
| 回归 Dashboard | `uv run light-regression dashboard` | `open regression_dashboard.html` |

## 关键约束

- `output/` 已 gitignore，用于本地验证和测试输出
- `data/` 已 gitignore，用于 Web 后端运行时数据（SQLite + 视频文件）
- 回归测试快照 `tests/regression/snapshots/` **禁止删除**
- 新 QC 规则必须零误报才提交
- light-qc 建议始终携带 `--transcript` 参数，以启用完整的时间轴对齐规则
- resume 见上文「断点续跑」；改 step 注册/hydrate 后跑 `tests/test_run_state.py`
