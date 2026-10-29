# 自主注意力（Autonomous Attention）

一个运行在 AstrBot 之上的会话级注意力插件。它把“是否该关注、是否该接话、什么时候主动去看看”从固定规则中抽离出来，交给一个可配置的兴趣决策模型；所有资源变更和发言许可仍由代码层批准。

## 核心模型

- **会话隔离**：兴趣、参与机会、检查点、关注/停止意图、消息进度、自主参与配额都以 AstrBot `unified_msg_origin`（UMO）为唯一所有权键。群号只用于路由和白名单，不能跨会话共享兴趣或检查点。
- **隔离会话路由**：启用成员隔离时，保留事件提供的真实群号与完整 UMO；同群不同 UMO 的兴趣和额度仍相互独立。尚未观察到的隔离 UMO 需要显式提供真实群号才能创建检查点。
- **兴趣（Interest）**：长期关注对象，带摘要、等待条件、证据消息、期限、剩余检查次数和无进展计数。默认每会话最多 3 个未关闭兴趣，默认有效期 120 分钟、检查余额 3 次。
- **参与机会（Opportunity）**：一次具体参与的证据锚点，必须引用本会话消息 ID、给出预计新增内容，并在有效期内未被消费。同一锚点跨主动/被动、换兴趣或换动作类型都只能获批一次。
- **检查点（Checkpoint）**：调度记录，分 `interest`（绑定兴趣）和 `session_scan`（管理员/种子扫描）。同一会话同一兴趣只保留一个未结束检查点；同一会话最多一个扫描检查点。
- **统一许可**：被动自主 `reply` 与主动 `participate` 都经过同一个控制合同：会话与硬策略 → 证据锚点 → 机会占用/过期 → 会话与全局自主配额 → 原子预留 → 生成/放行 → 提交结果。
- **显式请求优先**：私聊、@Bot、引用 Bot、唤醒前缀视为显式请求；硬策略通过后直接放行主模型，不占用兴趣槽位和自主配额。`unknown` 来源按自主路径保守处理。
- **关注不等于发言**：主 Agent 的 `focus_group` / `focus_topic` 只提高本会话兴趣优先级并记录证据，不再强制 `pass`；仍要经过证据和参与许可。

## 运行要求

- AstrBot `>= 4.26,<5`
- 无额外 Python 依赖

## 安装与启用

1. 将本目录放到 `AstrBot/data/plugins/astrbot_plugin_autonomous_attention`。
2. 在 WebUI 插件页重载插件。
3. 打开插件配置，至少设置：
   - `passive_mode`：建议先用 `block_only` 验证硬策略，再切到 `full`。
   - `decision_provider_id` / `main_provider_id`：兴趣模型与主动回复模型，留空使用当前会话模型。
   - `group_allowlist`：唯一群聊白名单；**留空表示不启用任何群聊注意力功能**。
   - `attention_limits`：兴趣容量、期限、检查余额、评估间隔和自主参与配额。
4. 先用 `/attention dryrun on` 观察决策，再关闭 dry-run。

> `group_allowlist` 是唯一群聊边界：能建立检查点的群才能接收主动消息；私聊只走被动判断，不受群白名单影响。

## 被动路径

```text
AstrBot 唤醒事件（私聊/@/引用/唤醒前缀）
  → 硬策略（用户忽略、白名单/黑名单、管理员覆盖）
  → 唤醒来源分类（private/mention/reply_to_bot/wake_prefix/unknown）
  → 显式请求：直接放行主模型（不走兴趣模型、不占自主配额）
  → 自主来源：会话节流 → 全局模型预算 → 兴趣模型一次
  → 机会证据校验 → 统一参与许可
  → pass（获批）或 observe/ignore（拒绝）
```

- 显式请求：`private`、`mention`、`reply_to_bot`、`wake_prefix`。即使 `passive_mode=full` 也直接放行；`ignore_user` 等硬策略仍然优先。
- 自主来源：`unknown`（无法确认来源，保守处理）或 `ambient`。没有新外部消息、未到最小评估间隔、配额不足或重复事件时直接 `observe`，不调用模型。
- 自主建议 `reply` 时必须引用本会话证据锚点并说明新增贡献，否则拒绝；拒绝不会默认放行。
- `observe` 也可通过 `interest_update` 提交摘要、等待条件和新证据，准入后建立有限检查点。重复证据、Bot 自身回复和只改摘要都不能续期。
- 模型超时、网络错误、非法结构或插件调用预算耗尽时统一 `observe`（拦截主模型），不再 fail-open。
- `pass` 只代表交给 AstrBot 主模型，不代表平台发送成功；`on_llm_response` 只用于关联生成结果并消费已预留机会。

## 主动路径

1. 调度器按会话轮转选择到期检查点，原子领取（`running` + `run_id`），同会话不重入。
2. 校验所有权、状态、TTL、剩余检查、白名单和静默时段；无新消息时跳过模型并退避（`min_interval × 2**no_progress`）。
3. 有未评估外部消息或管理员强制检查时，读取本会话窗口（带稳定消息 ID），运行受限 Agent。
4. Agent 工具：
   - `read_messages`、`get_interest`
   - `schedule_next_check`、`defer_check`
   - `create_checkpoint`（只能引用本会话已准入兴趣）
   - `update_user_policy`
   - `record_interest`
   - `request_main_reply` / `participate`
   - `observe`、`skip`、`end_focus`
5. 最终 `participate` 必须附带机会证据；通过统一许可后原子预留机会与配额，再生成回复。
6. 生成前和发送前复核：新外部消息（stale snapshot）、停止意图、兴趣状态、预算。失败/空回复/发送失败/结果不明分别记录，且不自动重发同一锚点。
7. 发送成功消费机会；兴趣继续观察，不因自己发言续期或增加检查余额。

同会话到期任务合并一轮判断时，其他兴趣仍保留独立检查点。模型通过 `arguments.evaluated_interest_ids` 列出实际评估过的兴趣；未评估的兴趣不会被扣除检查余额或推进游标。历史数据库 ID 与实时消息 ID 分开标识，并优先合并到实时消息身份；没有可信时间的消息不能成为自主参与锚点。

## 主模型注意力工具

| 工具 | 作用 |
| --- | --- |
| `attention_query` | 查询当前会话的会话、兴趣、检查点、意图和最近决策；`all` 仅表示会话内全部类别 |
| `attention_set_intent` | `focus_group`、`focus_topic`、`ignore_user`、`stop_focus`、`schedule_check` |
| `attention_ignore_user` | 创建有时限的用户忽略策略 |
| `attention_clear_user_policy` | 解除用户忽略策略 |
| `attention_focus_topic` | 设置当前会话的话题关注（提高兴趣优先级，不强制放行） |
| `attention_stop_focus` | 停止当前会话中主 Agent 发起的关注/检查 |

工具结果是 JSON（`ok`/`code`/`message`/`data`）。模型不能指定其他会话的 UMO 或 ID；跨会话读取/修改会被拒绝。`focus_topic` 会给本会话准入一个兴趣并记录证据，但不会绕过证据与配额。

## 管理指令

所有指令需要管理员权限，默认作用于**当前会话**；`--all` 表示显式全局汇总，不能用于修改其他会话资源。

| 指令 | 作用 |
| --- | --- |
| `/attention status [UMO\|--all]` | 状态、兴趣数、机会数、预算、调度器和最近决策 |
| `/attention list [UMO\|--all]` | 检查点列表 |
| `/attention interests [UMO\|--all]` | 兴趣列表 |
| `/attention add [分钟] [原因]` | 为当前群创建扫描检查点 |
| `/attention add <UMO> [分钟] [原因]` | 为指定群创建检查点 |
| `/attention check <ID>` | 立即强制检查一次 |
| `/attention pause\|resume\|remove <ID>` | 暂停/恢复/取消检查点 |
| `/attention ignore <用户\|@成员> [时长] [原因]` | 设置忽略策略（`30m`、`2h`、`1d`、`永久`） |
| `/attention unignore <用户\|@成员>` | 解除当前群和全局忽略 |
| `/attention policies` | 列出忽略策略 |
| `/attention budget [UMO\|--all]` | 查看当天预算 |
| `/attention dryrun on\|off` | 切换 dry-run |
| `/attention help` | 显示帮助 |

别名：`/aa`、`/注意力`。

## 预算与限制

- **模型/语义检查预算**（全局）：`daily_model_call_budget`、`daily_active_check_budget`；只统计插件自身 gateway 调用，不覆盖 AstrBot 原始主模型的全部调用。
- **自主参与预算**：每会话 `session_autonomous_reply_budget` + 全局 `daily_autonomous_reply_budget`；获批尝试时原子扣减，失败不返还。显式请求不计入。
- **冷却**：`global_send_cooldown_minutes` 与 `session_autonomous_cooldown_minutes`，从获批尝试计时，失败不消除本次冷却。
- **无进展**：连续 3 次无进展、检查余额耗尽、TTL 到期、明确结束或被更高优先级兴趣替换时关闭兴趣并取消关联检查点。
- **容量**：槽位满时只有更高等级的新候选可替换最弱可替换兴趣；同等级保留原兴趣；管理员兴趣受保护。

## 配置分组

- `general`：开关、被动模式、`dry_run`、日志、时区、决策记录上限、并发。
- `models`：兴趣模型/主模型、人格、两套评判标准。
- `groups`：唯一群白名单与禁止群。
- `context`：窗口条数、字符上限、保留会话数。
- `scheduling`：静默时段、检查间隔、全局冷却、`retry_later` 延迟、种子检查点。
- `agent_limits`：主动 Agent 轮数、工具调用、检查点修改、主模型调用、超时和输出长度。
- `attention_limits`：兴趣容量/期限/检查余额、最小评估间隔、机会期限、每会话自主配额、会话冷却、全局自主配额。
- `budgets`：模型调用、主动检查、全局自主尝试预算，以及策略/意图时限。
- `safety`：`owner_override`。

`attention_limits.daily_autonomous_reply_budget` 和 `session_autonomous_cooldown_minutes` 的配置默认值为 `-1`，表示继承旧配置；无旧值时实际默认均为 10。显式设置新值（包括 0）优先。兼容旧平铺键、分组键及 AstrBot Schema 自动补齐；旧字段保留为隐藏兼容项，并输出一次兼容提示。

## 状态迁移与恢复

- 状态版本为 `2`。首次加载 v1 时先备份为 `attention_state.v1-<时间戳>.json`，再原子写入 v2；迁移失败不会覆盖原文件，插件启动会报错而非以空状态继续。
- v1 的 `topics` 转为所属 UMO 的 `interests`；带 `topic_id` 的检查点迁移为 `interest`，无关联的转为有限期 `session_scan`。
- 无法确定所有权的兴趣/意图会暂停并记录迁移原因，不会被任何会话读取为当前状态。
- 多个会话引用同一旧话题、路由冲突等歧义会隔离原记录；扫描检查点也会获得一次性的 TTL 与检查余额。无法读取的状态文件会备份并停止启动，保留原文件。
- 同会话同兴趣的多个未结束检查点合并，保留来源优先的一条，被取消 ID 记录在 `migration.checkpoint_aliases`。
- 重启时 `running` 检查点恢复为可重试状态；`reserved` 参与机会标记为 `uncertain`，不自动重试。
- 已迁移文件再次加载不会新增兴趣、返还预算或延长期限。

## dry-run

- 仍调用兴趣模型并消耗真实模型调用预算。
- 不写兴趣、机会、配额、检查点等真实控制状态；不拦截主模型，也不发送主动消息。
- 外部消息窗口和管理员真实命令照常生效。
- 兴趣准入、机会去重、自主额度和后续调度在独立影子状态中执行完整校验；重复轮询不会反复判断同一快照。配置切换会丢弃影子资源，不追发模拟机会。

## 数据与隐私

- 状态文件：`AstrBot/data/plugin_data/astrbot_plugin_autonomous_attention/attention_state.json`。
- 默认 INFO 日志记录会话、决策、稳定原因码和简短理由，不记录完整聊天内容、人格或完整提示词。
- 判断记录包含 `umo/interest_id/checkpoint_id/anchor_message_id/wake_source`、消息进度、模型建议、代码决定、拒绝码、预留 ID 和发送结果。
- 汇总使用持久化累计计数，区分外部消息、语义判断、建议参与、获批尝试、被动放行和主动成功发送，不受最近日志条数限制。明确生成失败、明确发送失败与发送结果不明分别记录；平台超时或未知异常不自动重试。
- 工具调用返回结果不会进入兴趣模型 prompt；`build_passive_request` 会过滤 `role=tool`、工具协议消息和工具结果内容块。
- 群聊内容始终作为不可信数据传入提示词，系统提示明确要求不要执行其中指令。

## 目录结构

```text
main.py                 插件入口、事件钩子、工具与管理指令
astrbot_adapter.py      唯一的 AstrBot 适配层：唤醒分类、上下文过滤、发送
attention/
  control.py            统一控制层：会话所有权、兴趣/检查点/机会、参与许可
  engine.py             被动/主动协调、消息进度、配额、恢复与审计
  passive.py            单轮被动建议
  active.py             受限主动 Agent 循环
  scheduler.py          检查点调度与任务生命周期
  policy.py             硬策略、预算、冷却、静默时段
  store.py              v2 JSON 状态、v1 迁移与原子写入
  models.py             领域对象、来源、唤醒来源、状态常量
  prompts.py            两套提示词、带消息 ID 的窗口格式化
  validation.py         JSON 提取、建议结构与动作校验
  settings.py           配置快照、边界与旧键映射
tests/                  核心回归、迁移、隔离与适配层测试
```

## 测试

```bash
python -m pytest -q
```

核心测试使用 FakeGateway/FakeContextProvider/FakeSender，不依赖真实模型或等待。适配层测试在 AstrBot 未安装时自动跳过。

## 已知边界

- 主动检查依赖群消息窗口；插件会观察白名单群消息并随状态持久化。AstrBot 自身群历史关闭时，重启前未持久化的部分可能缺失。
- AstrBot 在部分版本会在普通用户请求上也写入 `provider_request` extra；被动门使用 `on_waiting_llm_request` 的时间点区分插件构造请求与普通用户请求，并显式跳过 `cron` 合成事件。
- 唤醒前缀可能被 AstrBot 在钩子前裁剪；来源分类同时检查原始平台消息、At/Reply 组件和配置前缀，无法确认时使用 `unknown` 保守处理。
- 主动参与只在群聊生效；私聊仅支持被动判断和兴趣状态。
- 第三方 Agent Runner 的 `ProviderRequest` 可能缺少完整 contexts；此时被动建议使用可获得的消息与 prompt。
- 主模型工具依赖 Provider 的 Function Calling 支持；不支持工具时被动/主动仍可工作，只是主模型无法查询或下达意图。
