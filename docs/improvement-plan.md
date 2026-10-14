# 后训练评测改进规划

状态：M1 已实施，M2–M6 待实施。日期：2026-10-14。本文是实施依据，实施中可按实际情况修订。

## 1. 目标

服务于后训练迭代：每产出一个 checkpoint，用固定的评测套件跑 SWE-bench Verified 和 Terminal-Bench 2.0，看到进度，随时可停可续，结束后得到与 baseline 的逐题对比。

核心使用流程：

```bash
# 部署新 checkpoint 后
python -m swe_eval submit --model qwen35-sft-step2000 --suite small-v1
python -m swe_eval status            # 或打开本地网页
python -m swe_eval report qwen35-baseline qwen35-sft-step2000 --suite small-v1
```

保持不变：mini-SWE-agent + SWE-bench 官方 harness、Harbor Terminus-2 + 任务 verifier、两套隔离依赖环境、冻结任务清单、原始轨迹和评分证据。不改 agent 提示词、动作协议和上游评分逻辑。

不做：登录/多用户、分布式调度、自动部署或切换 checkpoint、自动 fallback 模型、新 agent、W&B/MLflow 集成、React 前端工程。

## 2. 要解决的问题

| 问题 | 现状 | 改进 |
|---|---|---|
| 换 checkpoint 麻烦且易串配置 | 模型 JSON 混合端点、身份和预算；`.env` 的 `EVAL_*` 可隐式覆盖；`EVAL_API_KEY` 优先于实例 key | 模型与套件分离；`.env` 只放凭证；覆盖只能显式传参 |
| 非 Qwen 模型受限 | 兼容 API 强制 `thinking=false`，Qwen 参数与 vLLM 绑定 | 模型配置自带 `extra_body` 与 reasoning 字段，thinking 与后端解耦 |
| 看进度要翻日志 | `io.execute` 结束后才写日志；无统一状态 | 每次运行维护 `status.json`，日志实时写入 |
| 停止要人工收尾，无法续跑 | 无队列和取消；中断可能遗留容器 | 单一后台 supervisor + 文件队列；停止/恢复 |
| 预算耗尽补丁丢失 | 非 Submitted 一律空提交 | 保存工作区 diff，评分策略显式选择 |
| 看不到上下文压力 | 只汇总累计 token | 逐请求记录输入/输出/reasoning token 和 finish reason |
| small 样本偏 | 按 ID 取前 10，集中于少数仓库 | 固定种子的多仓库样本、显式清单、完整 split |
| 对比只能两两且条件全等 | `comparison.json` 全字段相等才能比较 | 多 checkpoint 对比，列出条件差异 |

## 3. 配置

主 CLI 继续无第三方依赖，配置使用 TOML（标准库 `tomllib`）。只分两类：模型和套件。agent 由 benchmark 决定，不单独配置。

```text
configs/
  models/qwen35-base.toml          # 端点、模型族、采样、模板、reasoning 参数
  models/qwen35-sft-step2000.toml  # extends = "qwen35-base"，只改 model/checkpoint
  suites/smoke.toml                # 每套 1 题，验证链路
  suites/small-v1.toml             # 固定种子多仓库样本
  suites/full.toml                 # 完整 split
```

模型文件（字段示意）：

```toml
extends = "qwen35-base"        # 可选，单层继承
model = "Qwen3.5-9B-sft-2000"  # 远端 model ID
checkpoint = "/root/ckpt/sft/step-2000"   # 操作者声明，记录但不当作已验证
base_url = "http://127.0.0.1:8000/v1"
api_key_env = "QWEN_API_KEY"   # 只读这个变量；缺失即报错，无回退
server = "vllm"                # vllm 时额外校验 /v1/models 的 root 与上下文；other 跳过
context_length = 65536
chat_template = "models/templates/qwen35-9b.jinja"   # 可选，仅用于记录指纹
[sampling]
temperature = 0.6
top_p = 0.95
presence_penalty = 1.5
[extra_body]                   # 原样透传，Qwen 专用参数只写在 Qwen 配置里
top_k = 20
chat_template_kwargs = { enable_thinking = true }
```

套件文件定义任务清单、预算和评分策略：`benchmarks`、各 benchmark 的 manifest 名称、`max_tokens`、`max_turns`、`swe_agent_timeout`、`swe_score_timeout`、`api_timeout`、`swe_scoring = "submitted-only" | "final-workspace"`。

规则：

- 优先级：模型 ← extends 基础模型；套件独立；只有 CLI `--set key=value` 能覆盖，覆盖项写入快照。`.env` 只用于凭证和 `HF_TOKEN` 等环境变量，不再读取 `EVAL_*`。
- 校验 `max_tokens < context_length`，未知字段报错。
- 提交时把解析后的模型与套件配置冻结到 `runs/<run-id>/config.json`，后续修改配置文件不影响已排队或运行中的任务。
- 现有 `models/*.json` 手工迁移到新格式，`Config` 类改为从冻结快照构造，worker 接口基本不变。

## 4. 运行模型

一次 `submit` 针对“模型 × 套件”，每个 benchmark 生成一个 run，`run-id` 默认为 `<model>__<suite>__<benchmark>__<时间戳>`。

### 4.1 队列与 supervisor

- `.state/queue/` 下每个排队 run 一个 JSON 文件；单 GPU 场景全局串行，按提交顺序执行。
- `python -m swe_eval supervisor` 为唯一后台进程，用 `flock` 锁保证只有一个实例；`submit` 发现没有 supervisor 时自动后台启动。`submit --foreground` 在当前终端同步执行，便于调试。
- 状态以 run 目录为准，`.state/` 只是队列和 PID 记录，删除后可从 `runs/` 重建。不引入 SQLite。

### 4.2 逐题执行

将两个 benchmark 统一为逐题循环，每题结束立即写 `tasks/<id>/result.json`：

- SWE：沿用现有逐题流程。
- Terminal-Bench：改为每题一个 Harbor job（并发本来就是 1），以便逐题停止、恢复和汇总。`tb_results` 改为读取单题 job。

run 级 `status.json` 持续更新：状态（`queued / running / stopping / stopped / completed / failed / interrupted`）、当前题与阶段（`setup / agent / scoring`）、阶段开始时间、已完成/已评分/通过数、worker PID 与启动时间、最后心跳。

### 4.3 停止与恢复

- `stop <run-id>`：立即停止。写入停止请求，supervisor 向当前 worker 进程组发送 SIGTERM，宽限期后 SIGKILL；按记录的容器 ID / compose project 清理容器；不开始后续题和后续 run。
- `stop --after-current`：当前题完成（含评分）后停止。
- 被停止的题保存已有轨迹和候选补丁，标记 `stopped`，不进入成绩分母之外的任何统计。
- `resume <run-id>`：重新入队，只执行没有终态 `result.json` 的题；重跑的题旧产物移入 `attempts/<n>/`，不从 agent 中间轮次续跑。
- supervisor 启动时检查 `status.json` 中 PID 与启动时间；进程已不存在的 running run 标记 `interrupted`，等待显式 `resume`。
- 本地停止只能保证客户端断开，远端是否立即中止生成取决于服务端，不做额外承诺。

### 4.4 结果语义

逐题记录三项，互不混用：

- 终止原因：`submitted / turn_limit / timeout / context_limit / output_truncated / api_error / env_error / stopped`。
- 补丁：是否正式提交、是否有候选补丁（仅 SWE）。
- 评分：`not_scored / passed / failed / scoring_error`。

run 汇总给出“计划 n、已评分 x、通过 y”。仅当所有题都已评分且无 API/环境错误时给出 `resolved_rate`，否则为 null，现有 `summarize` 口径保持。

### 4.5 SWE 候选补丁

- `swe_agent` 在 `finally` 中、容器清理前执行 `git diff` 保存 `candidate.patch`，失败单独记录原因。
- `submitted-only`（默认，与现有口径一致）：只对正式提交评分。
- `final-workspace`：预算耗尽时也对候选补丁评分，结果带 `scored_from=candidate` 标签。
- 评分策略属于套件，两种策略的 run 不做受控对比。
- `regrade <run-id> --policy final-workspace` 生成旁路结果目录，不改写原结果。
- Terminal-Bench 不适用，继续由 verifier 判定。

## 5. 监控指标

在现有 `events.jsonl` 和 `metrics.py` 基础上补充，不改变已有指标口径（见 [metrics.md](metrics.md)）：

- 逐请求：prompt / completion / reasoning token、finish reason、耗时。逐题汇总最大单次输入、最大单次输出、上下文占用峰值 `max_prompt_tokens / (context_length - max_tokens)`。
- 终止原因分类按 4.4 细化，`context_limit` 与 `output_truncated`（finish_reason=length）分开计数。
- 日志改为边运行边写入 run 目录，写入时脱敏；脱敏处理按行缓冲，避免 key 被读取块切断后漏过。
- 不采集 GPU / vLLM metrics。

## 6. 任务集

| 套件 | SWE-bench Verified | Terminal-Bench 2.0 |
|---|---|---|
| `smoke` | 1 题，链路验证 | 1 题 |
| `small-v1` | 12 题：12 个仓库各 1 题，固定 seed | 10 题，固定 seed 随机抽取 |
| `full` | 完整 test split（500） | 完整数据集 |

- `prepare` 支持三种来源：显式 ID 清单、`n + seed`（SWE 可按仓库分层）、完整 split。生成的 manifest 记录 seed、数据集 revision 和实际 ID，冻结在 `profiles/`。
- 现有 `validate_manifest` 中“固定 1/10 题”和“ID 必须排序”的约束改为按 manifest 自身声明校验。
- Terminal-Bench 若 registry 提供类别元数据再考虑分层，首版不做。
- gold patch 仍只存在于 manifest，不进入 agent 输入、网页和导出报告。

## 7. 对比与报告

`report <model...> --suite <suite>` 按 benchmark 汇总同一套件下多个模型的最新完整 run（也可直接传 run 目录）：

- 前提：任务清单哈希、benchmark、评分策略、agent 依赖指纹一致，否则拒绝。
- 预算、采样、模板等其他差异不阻止对比，但在报告顶部逐项列出。同一模型族的 checkpoint 回归通常应完全一致，有差异时醒目标注。
- 内容：每个模型的通过数/成功率/token/轮数/耗时/终止原因分布；逐题矩阵（每题各模型通过与否），首个模型作为 baseline 标出新增通过和回退。
- 不完整 run 只展示已配对评分的题和覆盖数，不给成功率差值。
- 输出 `report.json`、`report.csv` 和单文件 `report.html`（内联 CSS，离线可打开）。
- 现有 `compare` 命令保留为两模型版本的 `report` 别名。

## 8. 本地网页

只做监控和浏览，所有写操作复用 CLI 同一套函数。不新增前端工程。

- `python -m swe_eval serve`，标准库 `http.server`，默认绑定 `127.0.0.1:8765`，无鉴权（只监听本机；需要远程查看时用 SSH 端口转发）。
- 单个静态 HTML + 原生 JS，每 3 秒轮询 JSON 接口；不使用 SSE。
- 页面：
  - 队列与 run 列表：模型、套件、benchmark、状态、进度、通过数。
  - run 详情：逐题状态、当前阶段与耗时、token 与上下文峰值、停止/恢复按钮。
  - 单题详情：模型消息与工具调用时间线（reasoning 折叠）、补丁 diff、评分失败测试、原始日志链接。
  - 对比：选择套件与模型，展示第 7 节的报告。
- 文件接口只允许读取 `runs/` 下登记的路径，拒绝路径穿越。网页进程退出不影响 supervisor。

## 9. 代码布局

在现有平铺模块上增量修改，不重组包结构：

```text
swe_eval/
  config.py      # 改：TOML 模型/套件加载、extends、--set、冻结快照
  runner.py      # 改：逐题循环（含 TB 单题 job）、status.json、取消检查点
  io.py          # 改：实时日志写入、进程组与容器记录
  worker.py      # 改：候选补丁、TB 单题、manifest 采样
  telemetry.py   # 改：逐请求 usage / finish reason
  results.py     # 改：结果三元语义、report 多模型对比
  queue.py       # 新：文件队列、supervisor、stop/resume
  report.py      # 新：report.json/csv/html
  web.py         # 新：http.server 接口 + 内联页面
configs/models/  configs/suites/
```

CLI：`submit / status / stop / resume / regrade / report / serve / supervisor`，保留 `doctor / prepare / verify-scorer`。旧 `run` 改为 `submit --foreground` 的别名。

## 10. 实施阶段

| 阶段 | 内容 | 验收 |
|---|---|---|
| M1 配置 | TOML 模型/套件、extends、显式覆盖、凭证只读指定变量；thinking 与 server 解耦；迁移现有两份模型配置 | 单测：继承与覆盖、未知字段、缺 key 报错、非 vLLM thinking 模型生成正确的 `extra_body`；`doctor --no-api` 通过 |
| M2 执行生命周期 | 逐题结果、TB 单题 job、`status.json`、实时日志、队列与 supervisor、stop/resume、SWE 候选补丁 | 用假 worker 测试：排队串行、两种停止、无遗留子进程、resume 跳过已完成题、supervisor 重启标记 interrupted；真实 smoke 一次停止与恢复 |
| M3 任务集与指标 | `small-v1`/`full` manifest、逐请求 usage 与上下文峰值、终止原因细分 | manifest 可复现（同 seed 同 ID）；SWE small 覆盖 12 仓库；假事件流测试指标 |
| M4 报告 | 多模型 `report`、HTML 导出、`regrade` | 构造的 fixture 上：兼容性拒绝/差异列出、不完整 run 不出成功率差值、HTML 离线可读 |
| M5 网页 | `serve` 与四个页面 | 本地启动后可看到运行中进度、停止按钮实际生效、路径穿越被拒 |
| M6 真实验收 | Qwen baseline 与一个后训练 checkpoint 跑 smoke，再按需 small-v1 | 两套 benchmark 评分、轨迹、用量完整，生成对比报告 |

顺序 M1 → M2 → M3 → M4 → M5 → M6。M1–M4 已覆盖后训练评测的主要需求，网页可延后。另一个 checkpoint 未就绪时，M6 只做 baseline，不宣称多模型接入已验收。

## 11. 测试

- 继续使用 `unittest`，保留现有测试。
- 开发期用本地假 OpenAI-compatible 服务和假 worker，不依赖 GPU 和 Docker；Docker/真实模型只在 M2 和 M6 做少量集成验证。
- 重点覆盖：配置隔离与凭证、日志跨行脱敏、停止与领取下一题的竞态、resume 幂等、候选补丁与正式提交区分、`regrade` 不改原结果、报告兼容性判定。
- 修改评分或指标口径时提升 `schema_version`；旧 run（`schema_version=2`）的 `report` 读取兼容，缺失字段显示为空。
