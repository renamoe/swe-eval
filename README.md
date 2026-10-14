# SWE 与 Terminal-Bench 模型评测

本地命令行工具，调用远端模型 API，编排 agent、Docker 执行环境、官方评分器和实验记录。用于自部署模型及后训练 checkpoint 的编码回归评测。

| Benchmark | Agent 与动作协议 | 评分器 |
|---|---|---|
| SWE-bench Verified | mini-SWE-agent，文本 bash 动作 | SWE-bench 官方 harness |
| Terminal-Bench 2.0 | Harbor Terminus-2，文本 JSON 动作 | 任务自带 verifier |

当前版本使用 CLI；Web 工作台、任务队列和通用配置拆分尚待实施，详见 [改进规划](docs/improvement-plan.md)。历史测试产物已清理，仓库不包含可引用的 benchmark 成绩。

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

## 配置模型

模型实例 JSON 保存 API 地址、远端模型 ID、checkpoint 和预算；凭证放在本地 `.env` 或进程环境变量中。

### 自部署 Qwen3.5-9B

现有预设为 [seetacloud-qwen35-9b.json](models/seetacloud-qwen35-9b.json)：本地 API `http://127.0.0.1:8000/v1`，thinking 开启，上下文 65,536，输出上限 8,192，20 回合，SWE agent 时间预算 900 秒，`presence_penalty=1.5`。这些是当前配置值，不代表正式评测或复现排行榜的推荐预算。

创建本机 `.env`：

```dotenv
EVAL_MODEL_INSTANCE=seetacloud-qwen35-9b
EVAL_API_KEY=EMPTY
VLLM_CHAT_TEMPLATE_FILE=models/templates/qwen35-9b.jinja
```

`EMPTY` 只用于当前未启用 API 鉴权的本地隧道端点。服务开启鉴权后使用实际密钥。

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

vLLM 预检核对模型 ID、模型路径、上下文及版本，并要求提供本地 chat template。服务端需启用适用的 reasoning parser；Qwen 使用 `--reasoning-parser qwen3 --generation-config vllm`。模板及版本需与实际服务一致。

### 其他 OpenAI-compatible API

使用通用示例创建忽略追踪的本地实例：

```bash
cp models/openai-compatible.example.json models/example.local.json
# 编辑 endpoint、model、checkpoint 和预算
```

`.env` 示例见 [.env.openai-compatible.example](.env.openai-compatible.example)：

```dotenv
EVAL_MODEL_INSTANCE=models/example.local.json
MODEL_API_KEY=填写该服务的密钥
```

当前兼容 API 模式要求 `thinking=false`，不发送 Qwen 专用参数，也不验证 vLLM 元数据。通用 thinking 接入属于待实施规划。

配置优先级：进程环境变量 > `.env`；显式 `EVAL_*` 覆盖模型实例字段。`EVAL_API_KEY` 非空时优先于实例的 `api_key_env`，切换服务时应清理旧的模型、地址、预算和凭证覆盖项。

单次选择实例可使用 `--model-instance <实例名或JSON路径>`，仍受上述覆盖规则影响。框架默认项见 [.env.example](.env.example)，模型实例可覆盖默认预算。

## 运行评测

以下 `doctor` 和 `run` 会实际调用模型：

```bash
uv run --locked python -m swe_eval doctor --output runs/preflight/doctor.json

# smoke：每套 1 题
uv run --locked python -m swe_eval run swebench --profile smoke --run-id baseline-swe-smoke
uv run --locked python -m swe_eval run tbv2 --profile smoke --run-id baseline-tb-smoke

# small：每套 10 题
uv run --locked python -m swe_eval run swebench --profile small --run-id baseline-swe-small
uv run --locked python -m swe_eval run tbv2 --profile small --run-id baseline-tb-small
```

`run-id` 必须唯一。当前并发为 1、每题 1 次尝试、Harbor 不重试 trial，自动上下文摘要关闭。预算由当前实例和显式覆盖决定；smoke 只验证执行链路。

首次运行自动准备任务，也可单独执行：

```bash
uv run --locked python -m swe_eval prepare swebench --profile small
uv run --locked python -m swe_eval prepare tbv2 --profile small
```

`smoke/small` 分别取按 ID 排序后的前 1/10 题，不是随机或分层样本。数据版本、任务内容和哈希保存在生成的 `profiles/`；后续实验复用同一清单。该目录包含 SWE gold patch，禁止提供给 agent。删除后重新准备会重新解析上游版本，因此正式对比应保留同一份清单。

仅验证评分器，不调用模型：

```bash
uv run --locked python -m swe_eval verify-scorer swebench --run-id gold-swe
uv run --locked python -m swe_eval verify-scorer tbv2 --run-id oracle-tb
```

## 结果与比较

实验产物保存在 `runs/<run-id>/`：

| 产物 | 内容 |
|---|---|
| `config.json`、`run.json`、`manifest.json` | 模型、预算和冻结任务 |
| `dependencies.json`、`environment.uv.lock`、`comparison.json` | 依赖与比较条件 |
| `doctor.json` | 模型和 Docker 预检 |
| `results.json/csv`、`summary.json/csv` | 逐题结果及汇总 |
| `failures.json`、`docker-oom-events.json` | 失败轨迹索引和本地 OOM 观察 |
| `tasks/<id>/agent/`、`tasks/<id>/scoring/` | SWE 轨迹与官方评分 |
| `harbor/job/` | TB 原生轨迹、verifier 和 reward |

指标定义见 [metrics.md](docs/metrics.md)。缺失数据保留 `null`，不当作零；未完成实验不报告完整成功率，模型解题失败与 API/环境错误分开记录。

选择另一个 checkpoint，用相同任务和设置生成新实验后比较：

```bash
uv run --locked python -m swe_eval compare runs/baseline-swe-small runs/checkpoint-swe-small
```

比较核对任务、agent、提示词、环境、模板、采样和预算。不同评测条件会被拒绝，不能直接把不同预算的结果当作受控模型比较。

目前尚无统一 Web 监控、持久队列或恢复入口。预算耗尽时 SWE 没有正式提交的修改会保存为空提交，外部强制停止也可能需要人工收尾；这些限制的改进方案在 [规划文档](docs/improvement-plan.md) 中。

## 开发与版本控制

```bash
uv run --locked python -m unittest discover -s tests -v
```

代码、文档、模型示例、chat template 和三套依赖锁应纳入 Git。以下内容只留本地：

- `.env`、实际 SSH 配置、密钥及 `remote*.txt` 凭据文件。
- `runs/`、生成的 `profiles/`、日志和未来的 `.state/`。
- `.venv/`、缓存、解释器安装目录和构建产物。
- `models/*.local.json` 等机器专用配置。

保留运行环境可避免重复安装。需要从零清理测试数据时，可在没有任务运行时删除 `runs/` 和 `profiles/`。迁移正式实验应单独备份这些目录及凭据，在新机器重新执行 `sync`。

项目结构：`swe_eval/` 为编排和指标，`environments/` 为隔离依赖，`models/` 为预设/示例，`scripts/` 为安装与连接工具，`tests/` 为离线及原生集成测试，`docs/` 为指标与后续规划。
