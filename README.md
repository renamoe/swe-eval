# SWE 与 Terminal-Bench 模型评测

本地命令行工具，调用远端模型 API，编排 agent、Docker 执行环境、官方评分器和实验记录。用于自部署模型及后训练 checkpoint 的编码回归评测。

| Benchmark | Agent 与动作协议 | 评分器 |
|---|---|---|
| SWE-bench Verified | mini-SWE-agent，文本 bash 动作 | SWE-bench 官方 harness |
| Terminal-Bench 2.0 | Harbor Terminus-2，文本 JSON 动作 | 任务自带 verifier |

当前版本使用前台 CLI，一次评一个模型；多模型报告和 Web 浏览尚待实施，详见 [改进规划](docs/improvement-plan.md)。历史测试产物已清理，仓库不包含可引用的 benchmark 成绩。

## 安装

评测端需要 Linux/WSL2、Docker、[uv](https://docs.astral.sh/uv/getting-started/installation/) 和网络；模型可以运行在另一台 GPU 主机。

```bash
uv python install 3.11 3.12
bash scripts/setup.sh sync
docker info
uv run --locked python -m swe_eval doctor --no-api
```

三套环境分别使用自己的 `uv.lock`：

| 环境 | 固定依赖 |
|---|---|
| 主 CLI，Python 3.11 | 无第三方运行依赖 |
| `environments/swebench`，Python 3.11 | mini-swe-agent 1.17.5、swebench 4.1.0、litellm 1.92.0 |
| `environments/tbv2`，Python 3.12 | harbor 0.24.0、litellm 1.92.0 |

常规安装使用 `sync`，只有主动升级依赖时才执行 `bash scripts/setup.sh lock`，审阅并提交全部锁文件。

## 配置

配置分两类，均为 TOML，提交时冻结到 `runs/<run-id>/config.json`：

| 目录 | 内容 |
|---|---|
| `configs/models/` | 端点、远端模型 ID、checkpoint、上下文、采样、`extra_body`、reasoning 校验、凭证变量名 |
| `configs/suites/` | benchmark、任务清单规格、`max_tokens`、回合与超时预算、SWE 评分策略 |

规则：

- 模型文件可用 `extends = "<name>"` 单层继承基础模型，表（如 `[sampling]`）按键合并。
- 只有 CLI `--set key=value` 能覆盖配置，值按 TOML 解析（`--set max_turns=5`、`--set sampling.temperature=0`），覆盖项写入快照。
- `.env` 只放凭证和工具环境变量（如 `HF_TOKEN`、`EVAL_SWE_REVISION`）；其余 `EVAL_*`、`VLLM_*` 会被忽略并提示。
- 凭证只读模型文件 `api_key_env` 指定的变量，缺失即报错，无回退；`api_key_env = ""` 表示端点无鉴权。
- 未知字段、`max_tokens >= context_length`、带凭证的 URL 等直接报错。
- `server = "vllm"` 时 doctor 额外核对 `/v1/models` 的 root、上下文和 `/version`；`server = "other"` 跳过。`reasoning = true` 要求响应中 reasoning 与正文分离，与 server 无关。
- `extra_body` 原样透传到请求体，Qwen/vLLM 专用参数（`top_k`、`chat_template_kwargs`）只写在 Qwen 配置里。

### 自部署 Qwen3.5-9B

基础配置为 [qwen35-base.toml](configs/models/qwen35-base.toml)：本地 API `http://127.0.0.1:8000/v1`，thinking 开启，上下文 65,536，`presence_penalty=1.5`。当前隧道端点未启用鉴权，`api_key_env` 为空；服务开启鉴权后填写变量名并在 `.env` 中设置。

后训练 checkpoint 参考 [qwen35-sft.example.toml](configs/models/qwen35-sft.example.toml)，只改 `model`、`checkpoint`、`model_path`。不想纳入版本控制的配置命名为 `*.local.toml`。

SSH 连接信息使用忽略版本控制的本地文件：

```bash
cp scripts/qwen35-ssh.conf.example scripts/qwen35-ssh.conf
# 编辑 qwen35-ssh.conf，填写 SSH 主机、端口和密钥路径
bash scripts/qwen35-tunnel.sh start
bash scripts/qwen35-tunnel.sh status
```

隧道脚本使用别名 `seetacloud-qwen35` 和密钥认证；主机更换时同步更新本地 SSH 配置。`ControlPath` 使用主机相关 socket，避免复用旧连接。可以通过 `QWEN35_SSH_CONFIG` 指定其他配置文件。

```bash
# OpenAI SDK 模型列表与最小非 thinking 请求，会调用模型
# 结果写入 runs/qwen35-api-minimal/
environments/swebench/.venv/bin/python scripts/qwen35-api-smoke.py

# 需要时停止隧道
bash scripts/qwen35-tunnel.sh stop
```

vLLM 服务端需启用适用的 reasoning parser；Qwen 使用 `--reasoning-parser qwen3 --generation-config vllm`。`chat_template` 只用于记录指纹，需与服务端实际模板一致；提交后模板文件变化会被拒绝。

### 其他 OpenAI-compatible API

参考 [openai-compatible.example.toml](configs/models/openai-compatible.example.toml) 创建 `configs/models/<name>.local.toml`，在 `.env` 中设置其 `api_key_env` 指定的变量（示例为 `MODEL_API_KEY`）。reasoning 模型设 `reasoning = true`，供应商参数（如 `reasoning_effort`）写入 `[extra_body]`。

## 运行评测

以下 `doctor` 和 `run` 会实际调用模型：

```bash
uv run --locked python -m swe_eval doctor --model qwen35-base --output runs/preflight/doctor.json

# 依次运行套件中的全部 benchmark
uv run --locked python -m swe_eval run --model qwen35-base --suite smoke

# 只跑一个 benchmark，可指定 run-id
uv run --locked python -m swe_eval run swebench --model qwen35-base --suite small-v1 --run-id baseline-swe-small
```

`run` 在前台执行，每个 benchmark 一个 run，默认 `run-id` 为 `<model>__<suite>__<benchmark>__<时间戳>`。终端每题输出一行结果和耗时；子进程日志（`agent.log`、`harness.log`、`harbor.log`）边运行边写入，可以 `tail -f`。当前并发为 1、每题 1 次尝试、Harbor 不重试 trial，自动上下文摘要关闭。预算由套件和显式 `--set` 决定；smoke 只验证执行链路。

按 Ctrl-C 停止：当前子进程组先收到 SIGTERM，宽限期（SWE agent 60 秒、Harbor 120 秒）内保存轨迹和候选补丁，超时 SIGKILL；随后删除本 run 的容器（agent 容器的 `swe_eval.run` 标签、评分容器名、Harbor compose project），记录到 `cleanup.json`。未完成的题标记 `stopped`，汇总照常写出（`complete=false`、`stopped=true`），后续 benchmark 不再开始。清理期间再次 Ctrl-C 会被忽略。不支持恢复，重测时用新的 run 从头跑。本地停止只保证客户端断开，远端是否立即停止生成取决于服务端。

SWE 评分策略由套件的 `swe_scoring` 决定。默认 `submitted-only` 只对正式提交评分；`final-workspace` 在预算耗尽时对最终工作区 diff（`candidate.patch`）评分，结果带 `scored_from=candidate`，终止原因仍为 `budget_exhausted`。两种策略的 run 不能直接比较。

### 任务集

| 套件 | SWE-bench Verified | Terminal-Bench 2.0 |
|---|---|---|
| `smoke` | 1 题（显式 ID），链路验证 | 1 题 |
| `small-v1` | 12 题，12 个仓库各 1 题，固定 seed | 10 题，固定 seed |
| `full` | 完整 test split（500） | 完整数据集 |

套件的 `[manifests.<benchmark>]` 用三种方式之一声明清单：`ids = [...]`；`sample = n` 加 `seed`（SWE 可加 `stratify = "repo"`，先保证每个仓库一题再轮转）；`all = true`。抽样按 `sha256(seed:id)` 排序，与 Python 版本无关，同 seed 得到同一组 ID。

首次运行自动准备，也可单独执行：

```bash
uv run --locked python -m swe_eval prepare --suite small-v1
uv run --locked python -m swe_eval prepare swebench --suite full
```

清单冻结在 `profiles/<benchmark>-<name>.json`，记录规格、数据集 revision（TB 为任务 Git commit）和实际 ID，后续实验复用。修改规格必须换新的 `name`，否则拒绝运行。删除后重新准备会重新解析上游版本，正式对比应保留同一份清单。该目录包含 SWE gold patch，禁止提供给 agent。

仅验证评分器（不调用模型，仍需 `--model` 以记录配置）：

```bash
uv run --locked python -m swe_eval verify-scorer --model qwen35-base --suite smoke
```

## 结果与比较

实验产物保存在 `runs/<run-id>/`：

| 产物 | 内容 |
|---|---|
| `config.json`、`run.json`、`manifest.json` | 冻结的模型与套件配置、覆盖项和任务 |
| `dependencies.json`、`environment.uv.lock`、`comparison.json` | 依赖与比较条件 |
| `doctor.json` | 模型和 Docker 预检 |
| `results.json/csv`、`summary.json/csv` | 逐题结果（含终止原因、上下文峰值）及汇总 |
| `failures.json`、`docker-oom-events.json`、`cleanup.json` | 失败轨迹索引、本地 OOM 观察和结束时清理的容器 |
| `tasks/<id>/agent/`、`tasks/<id>/scoring/` | SWE 轨迹、`requests.json`（逐请求用量）、`candidate.patch` 与官方评分 |
| `harbor/job/` | TB 原生轨迹、`requests.json`、verifier 和 reward |

指标定义见 [metrics.md](docs/metrics.md)。缺失数据保留 `null`，不当作零；未完成实验不报告完整成功率，模型解题失败与 API/环境错误分开记录。

选择另一个 checkpoint，用相同任务和设置生成新实验后比较：

```bash
uv run --locked python -m swe_eval compare runs/baseline-swe-small runs/checkpoint-swe-small
```

比较核对任务、agent、提示词、环境、模板、采样和预算。不同评测条件会被拒绝，不能直接把不同预算的结果当作受控模型比较。

多模型报告与 Web 浏览尚待实施，见 [规划文档](docs/improvement-plan.md)。

## 开发与版本控制

```bash
uv run --locked python -m unittest discover -s tests -t . -v
```

代码、文档、模型示例、chat template 和三套依赖锁应纳入 Git。以下内容只留本地：

- `.env`、实际 SSH 配置、密钥及 `remote*.txt` 凭据文件。
- `runs/`、生成的 `profiles/`、日志和未来的 `.state/`。
- `.venv/`、缓存、解释器安装目录和构建产物。
- `configs/**/*.local.toml` 等机器专用配置。

保留运行环境可避免重复安装。需要从零清理测试数据时，可在没有任务运行时删除 `runs/` 和 `profiles/`。迁移正式实验应单独备份这些目录及凭据，在新机器重新执行 `sync`。

项目结构：`swe_eval/` 为编排和指标，`environments/` 为隔离依赖，`configs/` 为模型与套件配置，`models/templates/` 为 chat template，`scripts/` 为安装与连接工具，`tests/` 为离线及原生集成测试，`docs/` 为指标与后续规划。
