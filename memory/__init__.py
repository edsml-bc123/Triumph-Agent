"""
triumph-agent 记忆系统模块 (Memory Package)
===========================================
对外统一暴露：
- 数据模型：MemoryRecord, MemoryType, MemoryScope
- 物理存储与索引：MemoryStorage, parse_frontmatter, format_document, memory_slug
- 算法与管线管理：MemoryManager, should_store_memory, extract_json_array
- 切面插件：MemoryHook
"""

from memory.manager import (
    MemoryHook,
    MemoryManager,
    extract_json_array,
    normalize_text,
    should_store_memory,
)
from memory.models import CandidateMemory, MemoryRecord, MemoryScope, MemoryType
from memory.storage import (
    MemoryStorage,
    format_document,
    memory_slug,
    parse_frontmatter,
)

__all__ = [
    "CandidateMemory",
    "MemoryRecord",
    "MemoryType",
    "MemoryScope",
    "MemoryStorage",
    "MemoryManager",
    "MemoryHook",
    "parse_frontmatter",
    "format_document",
    "memory_slug",
    "should_store_memory",
    "normalize_text",
    "extract_json_array",
]
