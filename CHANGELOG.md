# 更新日志

仓库：<https://github.com/qwaecd/astrbot_plugin_autonomous_attention>

## 维护约定

- 发布前在文件顶部新增版本小节，标题格式为 `## vX.Y.Z - YYYY-MM-DD`，与 Git 标签一致。
- 变更按 `新增` / `变更` / `修复` / `移除` / `兼容性` 分类；面向使用者描述可感知的行为，不堆砌内部重构细节。
- 未发布的改动先写入 `## 未发布` 小节，发布时改名为版本号并补上日期。
- 发布时同步三处版本号：本文件、`metadata.yaml` 的 `version`、Git 标签。

## v1.2.0 - 2026-10-29

被动路径的行为与可观测性更新：群聊 @Bot、引用 Bot 改为交给兴趣模型判断，所有判定点与注意力模型调用都能从 INFO 日志追踪，`/attention status` 可直接看到活跃指令。

### 新增

- `/attention status` 增加“活跃指令”段落，列出 `stop_focus`、`focus_topic`、`schedule_check`、`ignore_user` 等指令的 ID、目标、来源、到期时间和原因；最近决策固定在消息末尾。
- 被动路径每个判定点输出 INFO 日志 `Passive attention gate: ...`，包含 `umo`、`user`、`wake_source`、`verdict` 和拒绝码；调用兴趣模型前输出 `verdict=invoke`，结束后输出 `Passive attention: ...`。
- 直接放行输出 `Passive explicit pass: ...`；主动路径调用注意力 agent 前输出 `Active attention agent invoked: ...`；适配层每次决策模型调用输出 `Attention decision model call: ...`。
- 新增回归测试：`mention` / `reply_to_bot` 走兴趣模型、`wake_prefix` 直接放行、状态快照展示活跃指令、意图时限新旧键兼容以及兴趣 TTL 不被较短的意图上限截断。

### 变更

- 群聊中的 `mention`（@Bot）与 `reply_to_bot`（引用 Bot）不再直接放行主模型，改为与 `unknown` 一样交给兴趣模型判断：需要证据锚点、占用会话与全局自主配额，未获批时拦截主模型；门控自身异常时也不再 fail-open。
- `private` 与 `wake_prefix` 仍保持直接放行，不调用兴趣模型、不占自主配额。
- 指标口径随之调整：`显式放行` 只统计 `private` / `wake_prefix`，@Bot 与引用 Bot 计入 `语义判断` 和自主配额。
- 主 Agent 意图时限改为分钟：`budgets.main_agent_max_intent_hours` 更名为 `main_agent_max_intent_minutes`，默认 30 分钟（原默认 24 小时）。停止关注、关注、话题关注和定时检查均使用该时限，停止关注后的静默上限也从 24 小时缩短为 30 分钟。
- 兴趣有效期仍由 `attention_limits.interest_ttl_minutes`（默认 120 分钟）决定，不会被较短的意图时限截断。

### 修复

- 唤醒来源判定与钩子日志改用 `getattr` 读取事件字段，适配器事件缺少 `unified_msg_origin` / `get_group_id` 时不再抛 `AttributeError`。
- 配置哨兵值 `-1` 不再被误裁剪为 1 分钟：缺少新键时回落默认 30 分钟，或按旧小时键换算。

### 兼容性

- 旧键 `budgets.main_agent_max_intent_hours` 仍可读取：缺少新键时按小时 × 60 换算为分钟并输出一次兼容提示；显式设置 `main_agent_max_intent_minutes` 时以新键为准。
- 行为变化会影响群里 @Bot、引用 Bot 的响应率：这些消息现在受 `passive_mode`、群白名单、评估节流（`min_evaluation_interval_seconds`）、静默时段和自主配额约束，配额耗尽时直接跳过。升级前建议先用 `/attention dryrun on` 观察，或调大 `attention_limits` 中的配额。

## v1.1.1 - 2026-10-29

### 变更

- `metadata.yaml` 新增 `repo` 字段，指向本仓库，插件市场可直接跳转；版本号更新为 `v1.1.1`。

### 修复

- 日志记录器统一使用 `from astrbot.api import logger`，移除 Python 内置 `logging` 的用法，符合 AstrBot 插件上架规范。

## v1.1.0 - 2026-10-29

「会话隔离与兴趣准入」重构，设计说明见 `vibe/attention-refactor-implementation.md`。

### 新增

- **会话隔离**：兴趣、参与机会、检查点、关注/停止意图、消息进度和自主参与配额，全部以 AstrBot `unified_msg_origin`（UMO）为唯一所有权键；群号只用于路由和白名单，不能跨会话共享兴趣或检查点。
- **兴趣（Interest）**：带摘要、等待条件、证据消息、期限、剩余检查次数和无进展计数的长期关注对象。默认每会话最多 3 个槽位、有效期 120 分钟、检查余额 3 次；`watching` / `paused` 都占用槽位，槽位满时只有更高等级的候选可以替换最弱兴趣。
- **参与机会（Opportunity）**：一次参与必须引用本会话消息 ID、说明预计新增内容，并在有效期内未被消费；同一锚点跨主动/被动、换兴趣或换动作类型都只能获批一次。
- **统一参与许可**：被动 `reply` 与主动 `participate` 共用同一条许可链（会话与硬策略 → 证据锚点 → 机会占用/过期 → 会话与全局自主配额 → 原子预留 → 生成/放行 → 提交结果）。
- **唤醒来源分类**：新增 `classify_wake_source`，从消息组件与平台原始消息判定 `private` / `mention` / `reply_to_bot` / `wake_prefix` / `unknown` / `ambient`，显式请求直接放行主模型，不占兴趣槽位和自主配额。
- **配额与冷却**：每会话每日自主尝试上限（默认 6）、全局每日自主尝试上限、会话自主冷却与全局冷却；获批时原子扣减，失败不返还，显式请求不计入。
- **状态 v2 与迁移**：首次加载 v1 状态先备份为 `attention_state.v1-<时间戳>.json`，再原子写入 v2；`topics` 转为所属 UMO 的 `interests`，无法确定归属的记录隔离并暂停，同兴趣的重复检查点合并，重启后 `running` 检查点恢复为可重试、`reserved` 机会标记为 `uncertain`。
- **dry-run 影子状态**：模拟模式下兴趣准入、机会去重、配额和调度在独立影子状态中完整校验，不写真实控制状态、不拦截主模型、不发送消息。
- **配置分组**：新增 `attention_limits`（兴趣容量/期限/检查余额、最小评估间隔、机会期限、每会话自主配额、会话冷却、全局自主配额）。
- **管理指令**：新增 `/attention interests`；`status`、`list`、`budget` 支持 `[UMO|--all]`。
- **平台限制**：`metadata.yaml` 增加 `support_platforms`，仅声明 `aiocqhttp`、`qq_official`、`qq_official_webhook`。
- **兴趣模型负反馈**：提示词展示候选兴趣的无进展次数与剩余检查、最近处理过的参与锚点及其结果、剩余会话/全局配额；主动检查无新消息时按 `min_interval × 2**no_progress` 指数退避。

### 变更

- 失败语义收紧：模型超时、网络错误、非法结构或预算耗尽统一 `observe`（拦截主模型），不再 fail-open 放行。
- 显式请求优先：私聊、@Bot、引用 Bot、唤醒前缀在硬策略通过后直接放行，即使 `passive_mode=full`；`unknown` 来源按自主路径保守处理。
- 关注不等于发言：主 Agent 的 `focus_group` / `focus_topic` 不再强制 `pass`，只提高本会话兴趣优先级并记录证据，仍须通过证据与配额校验。
- 检查余额只在真正评估过兴趣后扣除：模型须用 `arguments.evaluated_interest_ids` 列出实际评估过的兴趣，未评估的兴趣不扣余额、不推进游标。
- 主模型工具统一拒绝跨会话的 UMO 或 ID 读写；`attention_query` 的 `all` 仅表示会话内全部类别。
- 配置默认值调整：旧键 `group_send_cooldown_minutes`、`daily_active_message_budget` 默认改为 `-1` 并标记为隐藏兼容项，仅在缺少新键时作为回退。

### 修复

- 同一会话的到期检查点不再并发重入：调度器按会话轮转领取（`running` + `run_id`），同会话串行执行。
- 历史数据库消息 ID 与实时消息 ID 分开标识并优先合并为实时身份；没有可信时间的消息不能作为自主参与锚点。
- 生成前、发送前复核新外部消息（stale snapshot）、停止意图、兴趣状态与预算；失败、空回复、发送失败和发送结果不明分别记录，且不自动重发同一锚点。

### 兼容性

- 旧平铺键与旧分组键仍可读取，新键优先；首次读取旧键时输出一次兼容提示。
- v1 状态文件自动迁移且不覆盖原文件；迁移失败时启动报错，不会以空状态继续运行。
- `attention_limits` 中 `daily_autonomous_reply_budget` 与 `session_autonomous_cooldown_minutes` 的默认值为 `-1`，表示继承旧配置；旧配置无值时实际默认均为 10。
- 升级后建议在 WebUI 检查 `attention_limits` 各值，再关闭 dry-run。

## v1.0.1 - 2026-10-27

### 修复

- 修复主动注意力 Agent 永远不会被调用：AstrBot 部分版本会给普通用户请求也写入 `provider_request` extra，原先在 `event_message_type` 阶段判断会误跳过被动判断；改为在 `on_waiting_llm_request` 阶段标记插件构造的请求（`_aa_skip_passive`），并显式跳过 `cron` 合成事件。
- 被动请求的上下文过滤：丢弃 `role=tool` 的消息、带 `tool_calls` / `tool_call_id` 的消息以及 `tool_use` / `tool_result` 内容块，避免工具结果进入兴趣模型提示词。

### 测试

- 新增适配层回归测试，覆盖插件构造请求与普通用户请求的区分以及上下文过滤。

## v1.0.0 - 2026-10-27

首个发布版本。

### 新增

- **被动注意力**：在 AstrBot 唤醒 Bot 之后、主模型真正生成回复之前，判断这条消息是否值得交给主模型。
- **主动注意力**：按检查点时间读取有限的消息窗口，由受限 Agent 观察、调用工具，最后决定参与、跳过、继续观察或结束关注。
- **主从协作**：主模型是主 Agent，拥有明确意图的最高执行权；注意力 Agent 是后台执行单元，不得覆盖主 Agent 意图或管理员策略。
- **统一控制层**：主模型工具、被动路径和主动路径的用户策略、检查点与主 Agent 意图都经 `AttentionControlService` 校验后写入。
- **主模型按需工具**：`attention_query`、`attention_set_intent`、`attention_ignore_user`、`attention_clear_user_policy`、`attention_focus_topic`、`attention_stop_focus`；动态状态不注入主模型系统提示词。
- **硬性策略**：用户忽略、群聊白名单/黑名单、管理员覆盖、静默时段、单群/全局冷却、每日模型调用与主动发言预算，始终优先于兴趣模型输出。
- **结构化校验**：所有模型输出必须是合法 JSON；非法输出默认转为 `observe`，不会默认放行。
- **状态持久化**：检查点、话题、用户策略、主 Agent 意图、预算、决策记录和最近消息窗口写入 JSON，插件重启后恢复。
- **dry-run**：仍调用兴趣模型并记录决策，但不拦截主模型、不发送主动消息。
- **管理指令**：`/attention`（别名 `/aa`、`/注意力`），包含 `status`、`list`、`add`、`check`、`pause`、`resume`、`remove`、`ignore`、`unignore`、`policies`、`budget`、`dryrun`、`help`。
- **配置分组**：`general`、`models`、`groups`、`context`、`scheduling`、`agent_limits`、`budgets`、`safety`。
- 插件卸载时取消调度器、手动检查和模型调用任务。
