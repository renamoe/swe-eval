# 评测指标（schema_version=3）

每个模型实验保持每题 1 次尝试。`results.json/csv` 保存逐题指标，`summary.json/csv` 保存汇总，`failures.json` 索引失败任务的轨迹与评分日志。比较命令也输出对应指标差值。

| 指标 | JSON 字段 | 口径 |
|---|---|---|
| Resolved rate / pass@1 | `resolved_rate`, `pass_at_1` | 解决数 / 固定任务总数；1 次尝试时二者相同。存在 API／环境错误或计划任务缺失时为 null，不移除失败任务后重算分母。 |
| 平均工具调用轮数 | `avg_tool_rounds` | 对有观测的任务求平均；一条模型回复实际发送过至少一条 shell 命令算一轮，批量发送多条命令仍是一轮。包含成功与失败任务。另记 `avg_tool_call_count`（逐条命令数）、`avg_agent_rounds`（收到的模型回复数）。 |
| 输出 token | `output_tokens` | API 返回的 completion_tokens 总数，包含服务端报告的 reasoning token，不重复加 reasoning_tokens。另记 input_tokens；预检请求不计入 agent 用量。 |
| 工具格式错误率 | `tool_format_error_rate` | 原生解析器自动修复之前拒绝的次数 / 实际动作解析次数；跨任务采用总次数相除（micro），不平均逐题百分比。另记未修复错误与成功自动修复次数。仅被上游接受为 warning 的格式不计错误，原 warning 保留在事件中。 |
| Timeout 比例 | `timeout_rate` | 至少出现一次超时的任务数 / 固定任务总数；同时输出 `agent_timeout_rate`、`tool_timeout_rate`、`infrastructure_timeout_rate`。agent 预算超时不等同基础设施故障；API、环境准备和 verifier 超时归入基础设施项。 |
| OOM 比例 | `oom_observed_rate` | 本地 Docker OOMKilled／OOM 事件或明确 OOM 错误信号的任务比例。单独的 exit 137 不足以证明 OOM。明确 OOM 标记为环境错误，使 pass@1 不完整。托管 API 的内部 OOM 无法完全观测，此字段是已观测比例，不是无 OOM 的保证。 |
| 终止原因 | `termination`, `termination_counts` | agent 如何结束，与评分、INFRA 状态分开：`submitted / turn_limit / timeout / context_limit / output_truncated / api_error / env_error / stopped`。`submitted` 指正常结束并交给评分器；`final-workspace` 下预算耗尽的题仍按原终止原因记录。gold／oracle 无 agent，为 null。 |
| 上下文峰值 | `max_prompt_tokens`, `context_peak`, `avg_context_peak` | 单题最大单次输入 token；`context_peak = max_prompt_tokens / (context_length - max_tokens)`，即输入预算占用。汇总给出最大值和逐题平均。 |
| 逐请求用量 | `requests.json`, `max_completion_tokens`, `reasoning_tokens`, `length_finish_count`, `llm_seconds` | 每次模型请求的 prompt／completion／reasoning token、finish reason、耗时与错误类型。`length_finish_count` 统计 finish_reason=length（输出被 `max_tokens` 截断）的请求，与上下文超限分开。reasoning token 仅在服务端返回 `completion_tokens_details` 时可得，否则为 null。 |
| 失败完整轨迹 | `failures.json`, `trajectory_paths`, `events_paths`, `scoring_paths` | 保留原生轨迹、逐次完整模型输入输出、原始动作解析结果、实际工具调用及输出、终端 pane、错误和评分路径；不裁剪或采样失败任务。 |

mini-SWE-agent 的原始响应保存在 `llm_response` 事件中；Harbor 的 `LLMResponse` 不含 finish reason，观测层额外包装 `litellm.acompletion`，按 trial 写入 `llm_raw` 事件（只记 finish reason 与 usage）。

`*_tasks_observed` 说明每个汇总指标覆盖多少任务。没有观测时保留 null，不把缺失记录算成 0。`token_usage_complete` 标记逐题 token 用量是否覆盖全部观测到的模型调用；请求中断、API 缺 usage 时，已知 token 仍保存，但不能当作完整成本。

工具调用计数表示发送给执行环境的请求，包括开始执行后报错的调用；不包含 agent setup、verifier 命令和纯 completion 标记。格式正确但 shell 命令失败，是执行／解题问题，不是格式错误。`llm_call_count` 是框架级调用次数，不表示隐藏的 HTTP 传输重试次数。

mini-SWE-agent 使用原生 DefaultAgent、模型、环境与解析器，通过观测包装在每条消息之后原子保存原生轨迹。Harbor 使用原生 Terminus-2，通过观测包装保存每次 LLM 调用、解析与终端请求；原提示词、动作协议、重试策略和预算不变。`events.jsonl` 每个事件立即写入；Harbor 结束时另存 `messages.json`，包括 reasoning（若 API 提供）。

`trace_complete=false` 表示外部 kill／进程墙钟超时等导致终态未落盘：已收到的回复和已开始的动作仍全部保留，未返回的 API 回复无法补造。`metrics_complete=false` 表示部分指标不可获取，必须结合覆盖字段阅读；它与 `complete`（实验任务终态与基础设施完整性）分别报告。gold／oracle 不调用模型，agent 指标为不适用的 null。

单题 smoke 的 pass@1 只有 0% 或 100%，不能代表完整 benchmark 或统计稳定的成功率。不同回合、时间或输出预算的实验不应直接当作受控模型对比。
