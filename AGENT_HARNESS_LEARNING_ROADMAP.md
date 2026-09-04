# 🚀 AI Agent / Harness 架构师成长路线图与工程演进蓝图

> **核心哲学**：  
> **$\text{Agent Product} = \text{Model} + \text{Harness}$**  
> 不搞上层框架堆叠（拒绝 LangChain / CrewAI / AutoGen 等浅层 API 搬砖），坚持 **“底层源码精读 $\to$ 纯手写 Agent Runtime $\to$ 成熟 Harness 架构 Diff $\to$ 生产级 Agent Infra”** 的硬核工程跃迁路径。

---

## 目录
- [一、 战略愿景与核心学习方法论](#一-战略愿景与核心学习方法论)
- [二、 知识基底：learn-claude-code 全 17 章节核心技术全景](#二-知识基底learn-claude-code-全-17-章节核心技术全景)
- [三、 动手实战：triumph-agent 自研运行时 5 阶段演进计划](#三-动手实战triumph-agent-自研运行时-5-阶段演进计划)
- [四、 结对实操：如何利用 Coding Agent 保证 100% 学习效果？](#四-结对实操如何利用-coding-agent-保证-100-学习效果)
- [五、 进阶升维：Pi 与 DeepSeek Harness 源码架构对比研究](#五-进阶升维pi-与-deepseek-harness-源码架构对比研究)
- [六、 终局跃迁：Agent Engineering 与 Agent Infra 生产基建](#六-终局跃迁agent-engineering-与-agent-infra-生产基建)
- [七、 阶段推进时间表与架构师行动座右铭](#七-阶段推进时间表与架构师行动座右铭)

---

## 一、 战略愿景与核心学习方法论

### 1. 为什么拒绝“框架搬砖”，选择“手写 Runtime”？
* **框架的局限性**：LangChain、LangGraph、CrewAI 等外部框架封装了过多的黑盒抽象，API 频繁变动，且掩盖了底层的 **并发控制、上下文生命周期、Token 预算熔断、工具权限逃逸与状态机死锁** 等核心问题；
* **能力的本质**：真正的 Agent 架构师能力在于：**即便只给一个裸 LLM API，也能从零设计并实现一个能够长期自主运行、精准调用工具、自适应压缩上下文、多智能体物理隔离、可观测、抗崩溃与具备自愈能力的 Agent Runtime / Harness**。

### 2. 三步跃迁学习法
```text
 ┌───────────────────────────┐     ┌───────────────────────────┐     ┌───────────────────────────┐
 │   1. 拆解本质 (已完成)    │ ➔   │   2. 亲手造轮 (立即启动)  │ ➔   │   3. 架构对比 (进阶升维)  │
 │  深度精读 learn-claude-code │     │  从零手写 triumph-agent   │     │  精读 Pi & DeepSeek Harness│
 │      (s01 ~ s17)          │     │    (v0.1 ~ v0.5 演进)     │     │      (做 Architecture Diff)│
 └───────────────────────────┘     └───────────────────────────┘     └───────────────────────────┘
```

---

## 二、 知识基底：learn-claude-code 全 17 章节核心技术全景

本路线图建立在对 `learn-claude-code` 全部 17 个章节源码的地毯式精读与深度内化之上：

| 章节编号与名称 | 核心解决痛点 | 核心技术机制与架构原理 |
| :--- | :--- | :--- |
| **`s01_agent_loop`**<br>基础代理循环 | 如何让大模型具备自主思考与持续行动能力 | **双循环驱动骨架**：外部会话循环 + 内部 ReAct 执行循环；基于 `tool_use` 与 `tool_result` 的消息追加机制与终止边界判定。 |
| **`s02_tool_use`**<br>工具定义与分发 | 大模型如何安全地感知和修改外部世界 | **工具注册表与协议分发**：JSON Schema 强类型约束参数定义、工具派发路由表、执行异常安全捕获与语义反馈注入。 |
| **`s03_permission`**<br>权限审计与沙箱 | 防止大模型执行高危破坏性操作（如 `rm -rf /`） | **零信任权限拦截器**：规则匹配（Auto/Ask/Deny）、破坏性正则检查、交互式终端授权弹窗、工作区路径防逃逸沙箱。 |
| **`s04_hooks`**<br>生命周期拦截系统 | 如何在不侵入 Agent 核心逻辑的情况下横向扩展能力 | **切面事件总线（AOP）**：`UserPromptSubmit`, `PreToolUse`, `PostToolUse`, `Stop` 等生命周期钩子，支持日志记录、输出截断与安全注入。 |
| **`s05_todo_write`**<br>任务清单与状态机 | 大模型在长流程任务中容易遗忘目标或跑偏 | **结构化计划状态机**：维护显式 Todo 列表（Pending / In-Progress / Completed），每轮交互自适应刷新状态，引导大模型按部就班推进。 |
| **`s06_subagent`**<br>子智能体派生 | 复杂子任务会迅速污染主对话历史与上下文窗口 | **上下文物理隔离**：主 Agent 派生出专属 Subagent，分配纯净的新上下文（Fresh Context），仅将子任务最终结果回填主会话。 |
| **`s07_skill_loading`**<br>技能按需动态加载 | 所有的专家知识若全部硬编码在 System Prompt 中会导致 Token 爆炸 | **渐进式 Skill 发现机制**：初始化只加载技能元数据摘要，当且仅当相关任务被触发时，才按需 `view_file` 读取完整指令文档。 |
| **`s08_context_compact`**<br>上下文渐进式压缩 | 对话历史触碰 200k 极限时的 OOM 与失忆问题 | **5 层渐进式上下文压缩流水线**：Token 预算计算、微观修剪（Micro-Compaction）、大模型语义摘要（Summary Rollup）、429/529 容灾重试。 |
| **`s09_memory`**<br>长期记忆系统 | 跨会话偏好、历史教训与知识持久化 | **三层记忆架构**：Working Memory（当前任务）+ Session Memory（当前会话）+ Long-term Memory（全局持久化），自动提取与索引检索。 |
| **`s10_task_system`**<br>DAG 任务编排看板 | 复杂工程任务的前后依赖与并发调度 | **有向无环图（DAG）任务系统**：任务依赖关系拓扑排序、就绪任务并发调度、任务状态看板实时刷新。 |
| **`s11_background_tasks`**<br>后台异步长任务 | 避免编译、跑测试、起服务等长耗时命令阻塞主交互 | **异步子进程管理**：后台守护进程托管、标准输入/输出流管道捕获、Reactive 异步完成事件主动唤醒机制。 |
| **`s12_cron_scheduler`**<br>定时任务与保活 | 周期性巡检、轮询与心跳监控 | **两阶段 ACK / Rollback 定时调度器**：标准 5 字段 Cron 表达式解析、任务保活状态机、失败重试与优雅退出。 |
| **`s13_agent_teams`**<br>多智能体战队协同 | 多个 Agent 并发写代码时的 Git 冲突与代码踩踏 | **物理级环境隔离与通信总线**：基于 Git Worktree 的物理独立工作区、MessageBus 邮箱异步通讯、Plan Gate 方案审批门禁。 |
| **`s14_mcp_plugin`**<br>模型上下文协议插件 | 标准化集成第三方扩展服务与工具生态 | **MCP 协议标准化集成**：JSON-RPC 跨进程通信（STDIO）、动态工具发现、资源（Resource）与提示词模板（Prompt）解耦。 |
| **`s15_integrated_harness`**<br>终极集成操作系统 | 将上述所有子系统融合成高内聚、高并发的完整操作系统 | **工业级操作系统核心**：多线程事件循环与控制台线程解耦、全局 6 大并发锁（`agent_lock`, `team_lock`, `dag_lock` 等）拓扑图谱。 |
| **`s16_workflow_runtime`**<br>确定性工作流运行时 | 解决复杂业务在对话中不可控、不可重放、崩溃无法断点续跑 | **代码化编排（*The plan is code*）**：基于 SHA-256 语义 Hash 的 `WorkflowJournal` 日志断点恢复系统、轻量 Schema 强类型校验、红蓝对抗审查流水线。 |
| **`s17_goal_loop`**<br>目标驱动拦截闭环 | 解决大模型盲目自信、口头宣布胜利、半途而废的顽疾 | **目标未达誓不罢休（*不见铁证不撒鹰*）**：Worker + Evaluator 双模型分工、Stop 边界 6 大裁决拦截状态机、精准具象负反馈自愈驱动。 |

---

## 三、 动手实战：triumph-agent 自研运行时 5 阶段演进计划

在当前目录 `triumph-agent/` 下，从零手写自己的开源级 Agent Runtime。**严禁引入任何第三方 Agent 框架（如 LangChain）**，全面基于 **阿里云百炼（DashScope）OpenAI 兼容协议（`openai` SDK + `asyncio` + `pydantic`）** 构建。

### 🌟 阿里云百炼（DashScope）API 接入与模型选型矩阵

```text
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │ 【阿里云百炼 API 接入配置规范】                                                        │
 ├────────────────────────────────────────────────────────────────────────────────────────┤
 │ Base URL:       https://dashscope.aliyuncs.com/compatible-mode/v1                      │
 │ 环境变量:       DASHSCOPE_API_KEY="sk-..."                                            │
 │ Python 客户端:  from openai import AsyncOpenAI / OpenAI                                │
 ├────────────────────────────────────────────────────────────────────────────────────────┤
 │ 【分工模型选型 (双模型分工架构)】                                                      │
 │ 1. Worker (干活模型):    qwen3.8-flash                                                 │
 │                          (顶尖代码生成、Tool Calling 与极速响应能力)                   │
 │ 2. Evaluator (裁判模型): qwen3.6-flash                                                 │
 │                          (超快响应、专职在 Stop 边界做事实核查与拦截)                  │
 └────────────────────────────────────────────────────────────────────────────────────────┘
```

### 目录结构规划
```text
triumph-agent/
├── runtime/          # 核心执行引擎 (Loop, State, Event, Dispatcher)
├── context/          # 上下文生命周期与压缩引擎 (Budget, Compactor, Tokenizer)
├── tools/            # 工具协议与安全沙箱 (Registry, Bash, FileSystem, Git)
├── security/         # 权限审计与策略引擎 (Permission, Sandbox, DenyList)
├── memory/           # 三层记忆系统 (Working, Session, LongTerm)
├── orchestration/    # 多智能体与编排系统 (Subagent, Team, DAG, Worktree)
├── workflow/         # 确定性工作流与断点恢复 (Journal, Schema, Operators)
├── goal/             # 目标驱动闭环与裁判拦截 (Evaluator, StopHook)
├── client.py         # 阿里云百炼 API 客户端封装 (OpenAI 兼容模式)
└── main.py           # CLI 交互入口
```

### 演进阶段实施清单：

#### 🔹 阶段 1：V0 Mini Agent（3~5 天）—— 核心骨架搭建
* **目标**：用 < 300 行 Python 跑通基于阿里云 API 的自主 ReAct 循环；
* **核心模块**：
  1. `client.py`：基于 `AsyncOpenAI(base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", api_key=os.getenv("DASHSCOPE_API_KEY"))` 封装通用调用接口；
  2. `runtime/loop.py`：实现 `while True`，处理阿里云返回的 `tool_calls` ➔ 执行本地 handler ➔ 回填 `tool` 角色消息 ➔ 无工具调用时退出；
  3. `tools/registry.py`：编写 `read_file`、`write_file`、`bash`（带 120s 超时和 30KB 截断）；
  4. `runtime/state.py`：设计显式状态类 `AgentState(messages, total_tokens, status)`；
  5. `runtime/event.py`：落盘追加写 `runs/{timestamp}.jsonl` 记录每一步 Trajectory。

#### 🔹 阶段 2：V1 Safe & Observable Harness（1 周）—— 权限与切面
* **目标**：打造具备工业级安全审计与事件拦截的健壮底座；
* **核心模块**：
  1. `security/permission.py`：实现 `Auto / Ask / Deny` 策略；加入危险命令正则与目录穿越防逃逸沙箱；
  2. `runtime/hooks.py`：设计 `UserPromptSubmit`, `PreToolUse`, `PostToolUse`, `Stop` 钩子拦截体系；
  3. `context/budget.py`：引入 Token 预算硬上限管控。

#### 🔹 阶段 3：V2 Context & Memory Operating Engine（1 周）—— 上下文管理
* **目标**：解决长上下文爆炸与跨会话记忆留存；
* **核心模块**：
  1. `context/compactor.py`：实现 5 层渐进式压缩流水线（剔除历史 Tool 细节、语义摘要合并、保留最近轮次）；
  2. `memory/manager.py`：实现 Working / Session / Long-term 三层记忆持久化与按需注入。

#### 🔹 阶段 4：V3 Multi-Agent & Orchestration System（2 周）—— 多智能体并发
* **目标**：实现物理级安全的多任务并发与团队协作；
* **核心模块**：
  1. `orchestration/subagent.py`：Subagent 纯净上下文派生与防污染机制；
  2. `orchestration/dag.py`：有向无环图任务依赖调度与看板；
  3. `orchestration/worktree.py`：Git Worktree 物理独立工作区隔离；
  4. `runtime/background.py`：异步后台任务守护与主动唤醒。

#### 🔹 阶段 5：V4 Deterministic Workflow & Goal Loop（1~2 周）—— 终极闭环
* **目标**：融合确定性断点续跑与目标驱动自愈；
* **核心模块**：
  1. `workflow/journal.py`：基于 SHA-256 语义 Hash 的 JSONL 日志断点恢复系统（Resume 零 Token 开销）；
  2. `workflow/schema.py`：纯手写递归 JSON Schema 校验与容错提取器；
  3. `goal/evaluator.py`：基于阿里云 `qwen-turbo` 的独立裁判小模型与 Stop 边界 6 大裁决拦截器。

---

## 四、 结对实操：如何利用 Coding Agent 保证 100% 学习效果？

在 AI 辅助开发时代，**完全拒绝 Coding Agent 是低效的，但完全依赖 Coding Agent 是致命的（会导致“认知偷懒 / 假装掌握”）**。

最科学、最具学习爆发力的方式是：**“你担任首席架构师（Lead Architect），将 Coding Agent 作为你的结对副驾驶（Copilot）”**。

```text
 ❌ 错误做法（认知偷懒，毫无内化效果）：
 用户: "帮我用阿里云百炼 API 写一个完整的 Agent 系统。" 
 Agent: (生成 500 行代码) ➔ 用户: "跑通了，好厉害！" ➔ 结果: 关掉 AI 依然什么都不会写。

 🟢 正确做法（架构主导，深度内化闭环）：
 1. 架构契约先行: 你先亲手定义 State 数据结构、锁模型、模块依赖与接口签名。
 2. 渐进式协作:   每次只让 Agent 实现一个极简原子模块（如只写 bash.py 超时截断逻辑）。
 3. 极端边界追问: 对 Agent 生成的代码发起 4 大灵魂追问（模型抽风/OS报错/死锁/Token溢出）。
 4. 破坏性单测:   亲手写对抗单测（如故意传入非法 JSON、rm -rf / 命令），验证系统的自愈与拦截！
```

### 🌟 架构师四大结对实操守则

#### 守则 1：数据结构与契约必须由你主导（Contract First）
在写任何业务代码之前，**状态类（State）和函数签名（Interface）必须由你构思并先写出来**。
* 例如：在写 `runtime/state.py` 时，你要先自己决定：*“我的状态机需要包含 `messages`、`status`、`total_tokens`、`run_id` 哪些字段？”* 这一步逼迫你建立全局架构掌控力。

#### 守则 2：单点渐进式生成与地毯式 Review（Single-Unit Generation & Review）
* 每次只让 Agent 写一个独立的子函数（如：`_handle_tool_call` 的分发逻辑）；
* Agent 生成代码后，**你必须逐行看懂，能向自己解释清楚每一个 `try...except` 和状态迁移分支的用意**。对任何存疑的代码立刻追问到底。

#### 守则 3：主动发起“极端边界追问”（Adversarial Interrogation）
每写完一个模块，主动排查以下 4 个 Corner Case：
1. **大模型抽风了怎么办？**（如返回了非法 JSON、调用了不存在的工具名）；
2. **操作系统报错了怎么办？**（如命令超时、文件不存在、磁盘写满）；
3. **并发冲突怎么办？**（如多线程同时写同一个状态字典、同一个 `run_id` 被两个进程同时跑）；
4. **Token 爆炸了怎么办？**（如命令输出了 50 万字符的日志，会不会瞬间把上下文撑爆）。

#### 守则 4：亲手手写“破坏性单测（Adversarial Tests）”
**检验你是否真正掌握的标准，是看你能不能写出搞崩它的单测！**
* 阶段 1：写一个故意报错的 Bash 命令，看 Agent 能不能根据 `exit_code != 0` 自主纠错；
* 阶段 2：塞一个 `rm -rf /`，看你的 Permission 模块能不能铁面无私地弹窗拦截；
* 阶段 5：写一个故意不跑测试就口头说“我修好了”的 Mock 回复，看你的 Goal Evaluator 能不能在 Stop 边界一脚把它踢回去重写！

### 👥 黄金分工协作矩阵

| 开发环节 | 你的核心职责（首席架构师） | Coding Agent 的辅助职责（高效副驾驶） |
| :--- | :--- | :--- |
| **1. 架构设计** | 决定模块依赖、状态机流转图、锁模型、选型阿里云模型 | 补全标准 Pydantic Schema、格式化目录结构 |
| **2. 代码实现** | 主导核心循环（`while True`）、裁判拦截逻辑、业务契约 | 快速编写重复的样板代码（如 `subprocess.run` 细节、HTTP 封装） |
| **3. 极限排障** | 提出 Corner Case（死锁、上下文溢出、幻觉绕过） | 协助排查 Traceback 堆栈报错、提供重构优化建议 |
| **4. 验收沉淀** | 亲手执行 CLI 交互测试，记录设计决策与心得 | 编写自动化 Unit Test 模板与 Mock 工具桩 |

---

## 五、 进阶升维：Pi 与 DeepSeek Harness 源码架构对比研究

当你亲手完成了 `triumph-agent` 的核心实现后，带着踩过的坑与真实的工程痛点，去精读成熟开源 Harness 进行**架构 Diff 对比**：

```text
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │ 【三方架构 Diff 对比矩阵】                                                              │
 ├──────────────────────────┬──────────────────────────┬──────────────────────────────────┤
 │ 维度                     │ 自研 triumph-agent        │ DeepSeek Harness                 │
 ├──────────────────────────┼──────────────────────────┼──────────────────────────────────┤
 │ 架构核心思想             │ 显式状态机 + 模块化基座  │ Everything is a Plugin (Cordis)  │
 │ 工具集成范式             │ 本地 Registry + MCP      │ 插件即服务 (Plugin as a Service) │
 │ 上下文隔离               │ 物理 Worktree + Subagent │ 独立 Session 分叉与回放          │
 │ 事件驱动体系             │ AOP Hook 拦截器          │ 响应式 Event Bus 事件编排        │
 └──────────────────────────┴──────────────────────────┴──────────────────────────────────┘
```

### 1. 研究 Pi：
* 重点研究其轻量级调度器、最小化状态机设计与 Prompt 注入技巧；
* 对比：*“Pi 的扩展机制相比我的 Hooks 设计，优缺点分别是什么？”*

### 2. 研究 DeepSeek Harness：
* **Everything is a Plugin**：深入理解为什么它将 Model、Tool、Session、Sandbox、UI 统统抽象为插件；
* **Event-Driven Architecture**：研究如何从“函数硬编码调用”跃迁为“响应式事件总线”；
* **Trajectory Engineering**：研究其 Session Log 的恢复、分叉（Fork）、检索与回放（Replay）机制。

---

## 六、 终局跃迁：Agent Engineering 与 Agent Infra 生产基建

进入真正的企业级生产环境，关注重心将从“怎么写 Prompt”彻底转变为“系统的稳定性、成本、安全性与工程基础设施”：

```text
                     ┌──────────────────────────────────────────────┐
                     │            Agent Gateway / Server            │
                     └──────────────────────┬───────────────────────┘
                                            │ 鉴权 / 限流 / 路由
                                            ▼
                     ┌──────────────────────────────────────────────┐
                     │         Unified Agent Runtime Cluster        │
                     │  (triumph-agent Core: Loop, Memory, Context) │
                     └──────┬────────────────┬───────────────┬──────┘
                            │                │               │
                            ▼                ▼               ▼
                     ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
                     │ Docker / gVisor│ │ Session Store │ │ Evaluation   │
                     │ Tool Sandbox │ │ (Trajectory) │ │ Benchmark    │
                     └──────────────┘ └──────────────┘ └──────────────┘
```

1. **可靠性工程（Reliability & Fault-Tolerance）**：
   * 指数退避重试、多模型 Fallback 路由、网络闪断 Checkpoint 恢复、状态一致性回滚；
2. **评测体系（Evaluation Benchmark）**：
   * 基于真实 Trajectory 的自动化评测管线：任务成功率（Success Rate）、工具调用准确度、Token 效率指数、平均自愈轮次；
3. **安全与隔离（Production Sandboxing）**：
   * 基于 Docker / microVM / gVisor 的远程非可信代码安全执行沙箱。

---

## 七、 阶段推进时间表与架构师行动座右铭

| 阶段 | 周期 | 核心交付物 | 达成标志 |
| :--- | :---: | :--- | :--- |
| **阶段 1：V0 Mini Agent** | Day 1 ~ 4 | `loop.py`, `tools/`, `main.py` | 纯 Python 实现无框架自主修复单测 |
| **阶段 2：V1 Safe Harness** | Week 2 | `permission/`, `hooks/`, `budget/` | 拦截危险指令、完备的 Trajectory 日志 |
| **阶段 3：V2 Context & Memory** | Week 3 | `compactor/`, `memory/` | 200k 超长对话自适应压缩不失忆 |
| **阶段 4：V3 Multi-Agent** | Week 4 ~ 5 | `subagent/`, `dag/`, `worktree/` | 多任务 Git 物理隔离并发协同 |
| **阶段 5：V4 Deterministic** | Week 6 ~ 7 | `workflow/`, `goal/` | 代码化流水线断点续跑 + 裁判拦截自愈 |
| **阶段 6：Pi & DSH 源码精研** | Week 8 ~ 9 | 产出架构对比技术白皮书 | 彻底吃透成熟 Harness 设计精髓 |
| **阶段 7：Agent Infra 服务化** | Week 10+ | 生产级 Agent Server & Sandbox | 具备企业级 Agent 平台架构设计能力 |

---

### 💡 架构师座右铭
> **“Talk is cheap, show me the Harness.”**  
> 永远不要停留在使用别人的 Agent 框架，亲手构建属于你的 Agent 操作系统！
