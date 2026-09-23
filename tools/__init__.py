"""
triumph-agent 工具系统 (Tools Engine)
=====================================
核心设计与工具生态分布：
1. 核心容器与规范：
   - ToolRegistry: 100% 纯白板工具容器与防御性安全分发器 (Defensive Dispatcher)
   - ToolPlugin: 统一工具插件协议 (PEP 544 Protocol)，所有业务工具均满足此契约
2. 基础原子工具集 (Layer 2 基础层)：
   - BuiltinToolsPlugin: read_file, write_file, edit_file, glob, bash
3. 高层领域扩展工具集 (各业务领域高内聚管理，通过 register_plugin 统一装配)：
   - CompactTool (context/compactor.py): 主动上下文语义压缩工具 (compact_context)
   - SubAgentTool (orchestration/subagent.py): 物理上下文隔离的子智能体派生工具 (subagent)
   - BackgroundTaskTool (orchestration/background.py): 后台长任务管控工具集 (check_task, kill_task, list_tasks)
   - DAGTool (orchestration/dag.py): 有向无环图任务依赖调度工具集 (submit_dag, run_dag, get_dag_board)
"""

from .builtin import BuiltinToolsPlugin
from .registry import ToolPlugin, ToolRegistry

__all__ = [
    "BuiltinToolsPlugin",
    "ToolRegistry",
    "ToolPlugin",
]

