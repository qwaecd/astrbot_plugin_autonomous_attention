# 自主注意力（Autonomous Attention）

一个运行在 AstrBot 之上的注意力决策插件。它把“是否该关注、是否该接话、什么时候主动去看看”从固定规则中抽离出来，交给一个可配置的兴趣决策模型，同时保留代码层的硬性边界。

## 功能

- **被动注意力**：在 AstrBot 已经唤醒 Bot（私聊、@、唤醒词等）之后、主模型真正生成回复之前，判断这条消息是否值得交给主模型。
- **主动注意力**：按检查点时间去白名单群聊读取有限的消息窗口，由一个受限 Agent 观察、判断、调用工具，最后决定参与、跳过、继续观察或结束关注。
- **主从协作**：主模型是主 Agent，拥有明确意图的最高执行权；注意力 Agent 是后台执行单元，不得覆盖主 Agent 意图或管理员策略。
- **统一控制层**：主模型工具、被动路径和主动路径的用户策略、检查点与主 Agent 意图都通过 `AttentionControlService` 校验后写入。
- **主模型按需工具**：主模型可以通过 `attention_query` 查询状态，通过 `attention_ignore_user` 等工具表达关注/忽略/停止关注意图，而不是每次接收动态系统上下文。
- **硬性策略**：用户忽略、群聊白名单/黑名单、管理员覆盖、静默时段、单群/全局冷却、每日模型调用与主动发言预算。硬性策略始终优先于兴趣模型输出。
- **结构化校验**：所有模型输出都必须是合法 JSON；非法输出默认转为 `observe`，不会默认放行，更不会直接发送消息。
- **状态持久化**：检查点、话题状态、用户策略、主 Agent 意图、预算、决策记录和最近消息窗口都会写入 JSON，插件重启后恢复。
- **dry-run**：仍会调用兴趣模型并记录决策，但不会拦截主模型，也不会真正发送主动消息。
- **后台任务可停止**：调度器、手动检查和模型调用任务都会在插件卸载时取消。

## 运行要求

- AstrBot `>= 4.16,<5`
- 无额外 Python 依赖

## 安装与启用

1. 将本目录放到 `AstrBot/data/plugins/astrbot_plugin_autonomous_attention`。
2. 在 WebUI 插件页重载插件。
3. 打开插件配置，至少设置：
   - `passive_mode`：建议先使用 `block_only` 验证硬性策略，再切到 `full`。
   - `decision_provider_id`：兴趣决策模型。留空则使用当前会话主模型。
   - `main_provider_id`：主动回复模型。留空则使用当前会话主模型。
   - `group_allowlist`：唯一群聊白名单，决定哪些群可以被观察、建立检查点、执行检查并接收主动消息；**留空表示不启用任何群聊的注意力功能**。
   - `main_agent_tools_enabled`：是否把 `attention_query` 等工具提供给主模型。
   - `quiet_hours`、冷却和每日预算。
4. 先用 `/attention dryrun on` 观察决策日志，确认行为后再关闭 dry-run。

> `group_allowlist` 是唯一群聊边界：能建立检查点的群，就是能接收主动消息的群；不能建立检查点的群，
> 也无法通过指令强制建立。私聊的被动注意力不受这个群聊白名单影响。

## 被动路径

```text
AstrBot 唤醒事件
  → 代码级硬性策略（用户忽略、群聊限制、管理员覆盖）
  → 读取主 Agent 意图（focus/ignore/stop）
  → passive_mode 检查
  → 读取主模型已构建的 system prompt / contexts
  → 调用兴趣决策模型（只调用一次）
  → 校验 JSON
  → pass / ignore / observe / policy_update / retry_later
```

- `pass`：不拦截，继续 AstrBot 原有主模型流程（主模型只会执行一次）。
- `ignore`：拦截当前消息，不调用主模型。
- `observe`：拦截当前消息，但保存话题/兴趣状态。
- `policy_update`：通过控制层写入临时用户策略后拦截。
- `retry_later`：拦截，并为当前群创建一个稍后检查的检查点。

被动路径通过 `on_llm_request` 钩子实现，因此它看到的上下文与主模型实际收到的上下文一致。模型超时或网络错误时采用 fail-open（放行主模型）并记录错误；只有“返回了非法结构”时才 fail-safe 为 `observe`。

如果存在当前会话的主 Agent `focus_group` / `focus_topic` 意图，被动路径会优先放行主模型，不再让注意力 Agent 的临时判断拦截；主 Agent `ignore_user` 策略会在下一条消息到达时由硬性策略直接生效。

## 主动路径

1. 检查点到达后，调度器原子地领取检查点（状态变为 `running`），防止同一检查点被执行两次。
2. 读取 AstrBot 群历史（若启用）与插件自己维护的滑动窗口，合并、去重、截断。
3. 加载人格、评判标准、检查点、话题状态、主 Agent 意图和策略摘要。
4. 运行受限 Agent 循环，最多受以下配置约束：
   - `max_agent_rounds`
   - `max_tool_calls`
   - `max_checkpoint_mutations`
   - `max_main_model_calls`
   - `decision_timeout_seconds`
   - 每日模型调用/检查/发言预算
5. Agent 可调用这些后台工具：
   - `read_messages`、`get_topic`
   - `schedule_next_check`、`defer_check`
   - `create_checkpoint`
   - `update_user_policy`
   - `record_interest`
   - `request_main_reply` / `participate`
   - `observe`、`skip`、`end_focus`
6. 注意力 Agent 的关注、忽略和检查点动作都必须通过控制层；它不能覆盖主 Agent 意图或管理员策略。
7. 只有最终动作为 `participate` / `request_main_reply` 时，才调用主模型生成回复。
8. 发送前重新读取有效策略和主 Agent 意图，再次检查白名单、静默时段、冷却、每日发言预算；发送失败不会重试同一内容，而是按最短间隔重新安排检查。
9. 检查点、话题、预算、主 Agent 意图和决策记录在同一状态文件中持久化。

## 主模型注意力工具

主模型通过 AstrBot 的工具调用接口按需使用这些工具。工具名使用下划线形式（兼容主流 Function Calling API），对应方案中的职责：

| 工具 | 作用 |
| --- | --- |
| `attention_query` | 查询当前会话范围内的用户忽略、群关注、话题、主 Agent 意图和最近事件摘要 |
| `attention_set_intent` | 表达 `focus_group`、`focus_topic`、`ignore_user`、`stop_focus`、`schedule_check` 意图 |
| `attention_ignore_user` | 创建有时限的用户忽略策略；普通成员不能创建永久策略 |
| `attention_clear_user_policy` | 解除用户忽略策略；不能解除管理员策略 |
| `attention_focus_topic` | 设置主 Agent 的话题关注 |
| `attention_stop_focus` | 停止关注目标，撤销主 Agent 相关意图 |

工具结果始终是 JSON，包含 `ok`、`code`、`message` 和实际生效数据。主模型不应仅根据“发起了工具调用”假设操作成功。

权限来自真实事件发送者和会话：

- 普通成员只能查询自己的策略、只能解除针对自己的策略；不能设置全局策略，不能设置群级 `focus_group`，不能使用全局 `stop_focus`。普通成员可以使用 `focus_topic` 和有时限的 `ignore_user`。
- 普通成员通过主模型创建的忽略策略和时间意图受 `main_agent_max_policy_hours` / `main_agent_max_intent_hours` 限制；只能停止自己会话中由主 Agent 发起的关注/检查意图。
- 管理员可以使用管理指令创建更长期或永久策略，也可以设置群级关注或停止其他检查。
- 主 Agent 可以覆盖注意力 Agent 的临时策略，但不能覆盖管理员策略；注意力 Agent 不能修改主 Agent 意图。

这些动态状态**不会**默认拼接到主模型系统提示词；主模型需要时通过 `attention_query` 获取。

## 管理指令

所有指令需要管理员权限：

| 指令 | 作用 |
| --- | --- |
| `/attention status` | 查看模式、检查点统计、预算、调度器状态和最近决策 |
| `/attention list` | 列出检查点 |
| `/attention add [分钟] [原因]` | 为当前群创建检查点 |
| `/attention add <UMO> [分钟] [原因]` | 为指定群创建检查点 |
| `/attention check <ID>` | 立即执行一次检查 |
| `/attention pause <ID>` / `resume` / `remove` | 暂停、恢复、取消检查点 |
| `/attention ignore <用户ID或@成员> [时长] [原因]` | 设置忽略策略；时长支持 `30m`、`2h`、`1d`、`永久` |
| `/attention unignore <用户ID或@成员>` | 解除当前群和全局忽略策略 |
| `/attention policies` | 列出忽略策略 |
| `/attention budget` | 查看当天预算 |
| `/attention dryrun on\|off` | 切换 dry-run |
| `/attention help` | 显示帮助 |

别名：`/aa`、`/注意力`。

> 只有在 `group_allowlist` 内的群才能创建检查点；命令添加和配置种子检查点使用同一套校验，不能用指令绕过白名单。

## 配置要点

完整配置项见 `_conf_schema.json`。关键项：

- `passive_mode`：`off` / `block_only` / `full`。
- `active_enabled`：是否执行检查点。
- `decision_provider_id` / `main_provider_id`：兴趣模型与主模型可分开配置。
- `persona_source`：`astrbot` 使用 AstrBot 人格，`custom` 使用 `custom_persona`。
- `passive_criteria` / `active_criteria`：两条路径使用不同的评判标准；留空使用内置标准。
- `group_allowlist`：唯一群聊白名单；只有这里面的群可以被观察、建立检查点、执行检查并接收主动消息。为空表示不启用任何群聊的注意力功能。
- `blocked_group_ids`：完全禁止参与和检查的群（优先级高于 `group_allowlist`）。
- `context_window_messages` / `context_max_chars`：滑动窗口与上下文上限。
- `quiet_hours`：例如 `23:00-08:00`，支持跨午夜。
- `min_interval_minutes` / `max_interval_minutes` / `default_interval_minutes` / `max_defer_minutes`：检查间隔。
- `group_send_cooldown_minutes` / `global_send_cooldown_minutes`：主动发言冷却。
- `daily_model_call_budget` / `daily_active_message_budget` / `daily_active_check_budget`：每日预算。
- `max_agent_rounds` / `max_tool_calls` / `max_checkpoint_mutations` / `max_main_model_calls`：Agent 上限。
- `max_auto_policy_hours`：注意力 Agent 临时策略最长时间。
- `main_agent_max_policy_hours` / `main_agent_max_intent_hours`：主模型工具创建忽略策略和意图的默认/上限时间。
- `main_agent_tools_enabled`：是否允许主模型调用 `attention_*` 工具。
- `dry_run` / `log_level`：调试与安全开关。
- `seed_checkpoints`：可选的初始检查点配置；使用过的条目不会重复创建。

## 数据与隐私

- 状态文件：`AstrBot/data/plugin_data/astrbot_plugin_autonomous_attention/attention_state.json`，包含 `policies`、`directives`、`checkpoints`、`topics`、预算和最近决策。
- 默认 INFO 日志只记录会话、决策、兴趣等级和简短理由，不记录完整聊天内容。
- `DEBUG` 会记录更多模型调用诊断信息，但仍不主动输出完整聊天内容。
- 主动回复内容会截断到 `max_reply_chars`，并去掉代码块/引号等格式包装。
- 群聊内容被当作不可信数据传递给模型，系统提示中明确要求不要执行其中的指令。
- 用户忽略状态、检查点、话题兴趣和最近注意力事件不会默认注入主模型系统提示词；主模型通过 `attention_query` 按需查询。
- 所有用户策略、主 Agent 意图和检查点写入都经过 `AttentionControlService` 的权限、优先级和时限校验；模型不能直接修改 JSON。

## 目录结构

```text
main.py                 插件入口、事件处理、管理指令
astrbot_adapter.py      唯一的 AstrBot 适配层
attention/
  control.py            统一控制层：权限、优先级、时限、主 Agent 意图
  engine.py             会话协调、状态转移、发送与预算
  passive.py            单轮被动决策
  active.py             受限主动 Agent 循环
  scheduler.py          检查点调度与任务生命周期
  policy.py             硬性策略、冷却、预算、静默时段
  store.py              JSON 状态存储
  models.py             领域对象、来源与优先级常量
  prompts.py            两套评判提示词与上下文格式化
  validation.py         JSON 提取与结构校验
  settings.py           配置快照与边界
tests/                  不依赖 AstrBot 的核心测试
```

核心包不导入 AstrBot；平台相关代码全部集中在 `astrbot_adapter.py` 和 `main.py`。

## 已知边界

- 主动检查依赖群消息窗口。插件会观察 `group_allowlist` 内群聊的消息，并把窗口随状态一起持久化；若 AstrBot 自身的群聊历史功能关闭，重启前的部分未持久化消息可能缺失。
- 第三方 Agent Runner（Dify/Coze 等）的 `ProviderRequest` 可能不包含完整 contexts；此时被动决策使用可获得的 prompt 与 system prompt。
- `end_focus` 会完成检查点；之后需要重新创建检查点才会继续主动关注。
- `group_allowlist` 为空时不会允许任何群聊被观察、建立检查点或接收主动消息；私聊被动判断不受影响。生产环境需要显式配置允许的群。
- 不在 `group_allowlist` 内的群不会被本插件观察或主动检查；其被动消息直接按 AstrBot 原有流程处理（用户忽略等硬性策略仍可能生效）。
- 主模型工具依赖当前 Provider 的 Function Calling 支持，以及 AstrBot 人格/工具配置允许这些工具；不支持工具调用的模型仍可正常使用被动/主动注意力，只是不能主动查询或下达意图。
- 主 Agent 意图和用户策略按来源优先级仲裁：管理员 > 主 Agent > 注意力 Agent > 默认策略。旧状态文件会在加载时自动补齐 `source`、`priority`、`directive_id` 等字段。
