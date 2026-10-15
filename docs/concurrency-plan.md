# 题目级并发方案

状态：待评审。日期：2026-10-15。修订 [improvement-plan.md](improvement-plan.md) 第 4 节"单线程前台"。

## 1. 目标与边界

同一个 benchmark run 内同时跑 N 题（本机 16GB 内存，N 上限按 2 设计），墙钟时间约降到 1/N。仍是单模型、前台、Ctrl-C 只清理不恢复；benchmark 之间仍串行；不做跨 run 或跨模型并发。

默认 `concurrency = 1`，行为与现在完全一致。

## 2. 改动

| 位置 | 改动 | 规模 |
|---|---|---|
| `config.py` | 套件新增 `concurrency`（int，默认 1，范围 1–4），可 `--set concurrency=2` | ~5 行 |
| `io.py` | `execute` 把子进程登记到一个全局集合，结束时移除；新增 `kill_all()` 对登记的进程组直接 SIGKILL | ~15 行 |
| `runner.py` SWE | `run_swe` 的单题主体抽成 `swe_task()`，用 `ThreadPoolExecutor(N)` 调度（N=1 走同一路径）；`rows`/`results.json`/进度计数加锁；一个 `stopping` 事件，题目在 agent 结束后、评分前检查它 | ~60 行 |
| `runner.py` TB | `tb_config` 的 `n_concurrent_trials` 取 `concurrency`；`TrialWatcher` 和清理已按 trial 前缀处理多 trial，不改；去掉 `HARBOR_GRACE` | ~3 行 |
| `runner.py` 记录 | `comparison.json` 的 `concurrency` 由写死的 1 改为实际值 | 1 行 |
| 测试 | 见第 5 节 | ~120 行 |
| 文档 | improvement-plan 第 4 节、metrics.md 计时口径、README | 少量 |

不需要改：`report`（`concurrency` 不在兼容门控里，会作为条件差异自动列出）、网页（已按目录状态区分 queued/agent/scoring，多题同时运行可直接显示）、`regrade`（评分量小，保持串行）。

## 3. 关键设计

**线程而非进程。** 每题的 agent 和评分本来就是子进程，线程只负责等待，GIL 无影响，且共享同一个 `rows` 与清理逻辑。

**中断：只做"发现不对劲时立即放弃"。** 中断后的 run 不用于对比，需要重测时用新 run-id 从头跑，所以不为在跑的题保存任何东西：

1. 主线程等待所有 future，收到 Ctrl-C/SIGTERM 后置位 `stopping`，`shutdown(cancel_futures=True)` 丢弃排队题，再 `kill_all()` 直接 SIGKILL 所有子进程组，不给宽限期。
2. 工作线程的 `wait()` 随之返回，看到 `stopping` 即退出，不再评分。
3. 主线程走现有 `finally`：`docker rm -f` 本 run 的容器和 compose 网络（SIGKILL 后 agent 和 Harbor 不会自行清理，这一步兜底），未完成题标 `stopped`，汇总照常写出，`complete=false`。
4. 清理期间忽略 SIGINT，保留现有处理。

代价：在跑的题拿不到 `candidate.patch` 和最后一步轨迹（轨迹每步已落盘，只丢最后一步），这些题本来就是 `stopped`。超时处理不变：agent 墙钟超时仍走 SIGTERM 加 60s 宽限，worker 会保存轨迹和候选补丁，`final-workspace` 评分依赖它。

**并发安全。** 每题独立目录；SWE agent 容器带 run 标签，评分容器名 `sweb.eval.<instance>.<run_id>` 按题唯一，harness 的 `logs/` 和报告写在各题 `scoring/` 下，互不覆盖。共享写只有 `results.json` 和终端进度，统一用一把锁。`results.json` 中的题目顺序仍按 manifest，与完成顺序无关。

**进度输出。** 开始一行 `start <id>`，结束一行 `[完成数/总数] <id>: <状态> (<耗时>)`，按完成顺序打印。

## 4. 对结果口径的影响

- 通过/失败的判定逻辑不变；逐题 token、轮数不受影响。
- 受影响的是墙钟相关项：服务端同时处理 N 路时单路 decode 略降（显存带宽受限场景，N=2 预计下降 <10%，未实测），本机 CPU/IO 也有争用。agent 900s 墙钟上限因此更容易触发，逐题 `llm_seconds` 和 `elapsed_seconds` 偏大。
- 因此 `concurrency` 写入快照和 `comparison.json`，`report` 在顶部作为条件差异列出。同一组对比应使用同一并发度；耗时类指标不跨并发度比较。

## 5. 测试

用现有假 worker/假 OpenAI 服务，不依赖 Docker：

- N=2 时两题执行区间重叠，总耗时约为串行的一半；`results.json` 顺序与 manifest 一致。
- 两题运行中 Ctrl-C：两个进程组立即被 SIGKILL（含忽略 SIGTERM 的子进程），被杀的题不进入评分，排队题与在跑题都标 `stopped`，`complete=false`，清理函数调用一次。
- 单题失败（环境错误）不影响另一题继续完成。
- `concurrency` 校验：非整数、0、超过 4 报错；`tb_config` 的 `n_concurrent_trials` 跟随配置。
- `concurrency=1` 时现有测试全部不变通过。

真实验收：small-v1 以 `--set concurrency=2` 各跑一次 SWE 和 TB，期间 Ctrl-C 一次确认 `docker ps` 无残留，并对照 `docker-oom-events.json` 无 OOM。

## 6. 资源前提

- **本机内存**：WSL2 默认只给物理内存的一半，现在 Docker 可用约 7.6GiB。每个并发槽位峰值约 3–5GB（SWE agent 容器与评分容器不同时存在；TB 任务声明 `memory = "2G"` 左右，另有 Harbor 自身开销），N=2 建议在 `%UserProfile%\.wslconfig` 设 `memory=12GB`、`swap=8GB`，执行 `wsl --shutdown` 后生效（会中断正在运行的 WSL 进程和隧道）。
- **云端**：现有 `--max-num-seqs 16` 和约 277k token 的 KV 池足够 N=2，无需改动。
- **磁盘**：不变。

## 7. 工作量与收益

- 编码加单测约 3 小时，真实验收约 3 小时（small-v1 两个 benchmark 各一次）。
- 收益：small-v1 单模型预计从 4–5 小时降到约 2.5 小时；SWE 几乎全是模型时间，加速接近 2 倍，TB 略低于 2 倍。
