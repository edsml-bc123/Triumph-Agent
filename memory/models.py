"""
triumph-agent 长期记忆数据模型定义 (Memory Models)
===================================================
对标 learn-claude-code s09 规范：
采用强类型 (str, Enum) 与显式反序列化验证器，杜绝类型体操与冗余定义。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Set


class MemoryType(str, Enum):
    """
    长期记忆四大持久化类型
    """
    USER = "user"            # 用户的长期偏好
    FEEDBACK = "feedback"    # 跨任务适用的指导反馈
    PROJECT = "project"      # 稳定的项目事实
    REFERENCE = "reference"  # 外部查找线索与资源

    @classmethod
    def valid_values(cls) -> Set[str]:
        """获取所有有效类型字符串集合"""
        return {item.value for item in cls}

    @classmethod
    def from_str(cls, value: str) -> MemoryType:
        """安全转换字符串为 MemoryType 枚举，非法值优雅降级为 PROJECT"""
        try:
            return cls(value.strip().lower())
        except (ValueError, AttributeError):
            return cls.PROJECT


class MemoryScope(str, Enum):
    """
    记忆候选作用域：仅 PERSISTENT 允许跨会话持久化存储
    """
    PERSISTENT = "persistent"
    CURRENT_TASK = "current_task"

    @classmethod
    def valid_values(cls) -> Set[str]:
        return {item.value for item in cls}

    @classmethod
    def from_str(cls, value: str) -> MemoryScope:
        try:
            return cls(value.strip().lower())
        except (ValueError, AttributeError):
            return cls.CURRENT_TASK


@dataclass
class CandidateMemory:
    """
    从大模型自由输出中反序列化提取的非受信候选记忆实体
    通过 from_raw 保证字段完备性，杜绝下游隐式类型错误
    """
    name: str
    type: MemoryType
    scope: MemoryScope
    description: str
    body: str

    @classmethod
    def from_raw(cls, raw: object) -> Optional[CandidateMemory]:
        """
        数据清洗与类型安全门禁：
        校验原始输入是否具备完整字段，非法或缺失时返回 None
        """
        if not isinstance(raw, dict):
            return None

        name = str(raw.get("name", "")).strip()
        raw_type = str(raw.get("type", "")).strip().lower()
        raw_scope = str(raw.get("scope", "")).strip().lower()
        description = str(raw.get("description", "")).strip()
        body = str(raw.get("body", "")).strip()

        if not name or not description or not body:
            return None
        if raw_type not in MemoryType.valid_values():
            return None
        if raw_scope not in MemoryScope.valid_values():
            return None

        return cls(
            name=name,
            type=MemoryType(raw_type),
            scope=MemoryScope(raw_scope),
            description=description,
            body=body,
        )


@dataclass
class MemoryRecord:
    """
    单条记忆数据实体，对应 .memory/{slug}.md 文件结构
    """
    name: str
    type: MemoryType
    description: str
    body: str
    filename: str = ""
    scope: MemoryScope = MemoryScope.PERSISTENT

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "type": self.type.value,
            "description": self.description,
            "body": self.body,
            "filename": self.filename,
            "scope": self.scope.value,
        }
