# 🚀 Triumph-Agent

> **核心哲学**：$\text{Agent Product} = \text{Model} + \text{Harness}$  
> 不搞上层框架堆叠（拒绝 LangChain / CrewAI 等黑盒搬砖），基于阿里云百炼原生协议与纯 Python 手写的开源级自主 Agent Runtime / Harness。

---

## 🌟 当前版本 (v0.1 - V0 Mini Agent)

- **通信层 (`client.py`)**：基于原生 `httpx.AsyncClient`，长连接池复用，集成 `json_repair` 容错修复大模型畸变 JSON，零 OpenAI SDK 依赖；
- **状态机 (`runtime/state.py`)**：显式生命周期状态枚举 (`AgentStatus`)，协议规范的消息闭环追加，步数安全硬熔断，向前兼容 `s17 Goal Loop` 拦截机制；
- **工具沙箱 (`tools/registry.py`)**：内置 `safe_path` 路径防逃逸、`run_bash`（120s 超时与 50KB 输出物理截断）、`read_file`、`write_file`、`edit_file`、`glob`；
- **核心引擎 (`runtime/loop.py`)**：双循环解耦的自主 ReAct 执行内循环引擎，批量工具派发，高透明 `loguru` 结构化可观测性输出；
- **CLI 终端 (`main.py`)**：开箱即用的交互式外层会话入口。

---

## 🛠️ 快速开始

### 1. 安装依赖
```bash
pip install httpx loguru pydantic python-dotenv json-repair
```

### 2. 配置密钥
复制配置模板并填入你的阿里云百炼 API Key：
```bash
cp .env.example .env
# 编辑 .env 文件中的 ALIBABACLOUD_API_KEY
```

### 3. 启动交互式 Agent
```bash
python main.py
```

---

## 📂 项目结构

```text
triumph-agent/
├── .env.example          # 环境变量脱敏模板
├── AGENT_HARNESS_LEARNING_ROADMAP.md # 学习路线与架构蓝图
├── client.py             # 阿里云百炼原生 HTTPX 客户端
├── main.py               # 交互式 CLI 会话入口 (Outer Loop)
├── runtime/
│   ├── loop.py           # ReAct 自主执行循环引擎 (Inner Loop)
│   └── state.py          # 中央状态机与上下文模型 (State Contract)
└── tools/
    └── registry.py       # 工具注册中心与安全执行沙箱 (Tool Registry)
```
