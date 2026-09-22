"""
triumph-agent 记忆持久化存储引擎 (Memory Storage)
===================================================
对标 learn-claude-code s09 设计：
1. 文件规范：单条记忆对应一个 .memory/{slug}.md 文件，采用标准 YAML frontmatter + Markdown 正文；
2. 索引机制：维护 .memory/MEMORY.md 目录索引，供两阶段粗筛召回；
3. 安全审计：严格校验路径，防止跨目录与逃逸工作区攻击；
4. 原子事务：快照捕获与灾难回滚，保证整理过程崩溃时 100% 还原。
"""

import asyncio
import os
import re
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml
from loguru import logger

from memory.models import MemoryRecord, MemoryType


def parse_frontmatter(text: str) -> Tuple[dict, str]:
    """
    解析 Markdown 文本中的 YAML Frontmatter 与 Markdown 正文
    """
    if not text.startswith("---\n"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    try:
        metadata = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError:
        return {}, text
    if not isinstance(metadata, dict):
        return {}, text
    return metadata, parts[2].strip()


def memory_slug(name: str) -> str:
    """
    清洗记忆名称为合法安全的文件名 slug
    """
    slug = re.sub(r"[^\w]+", "-", name.lower()).strip("-_")
    return slug or "memory"


def format_document(name: str, mem_type: str, description: str, body: str) -> str:
    """
    将元数据与正文序列化为标准 Frontmatter Markdown 文档
    """
    metadata = yaml.safe_dump(
        {"name": name, "description": description, "type": mem_type},
        sort_keys=False,
        allow_unicode=True,
    ).strip()
    return f"---\n{metadata}\n---\n\n{body.strip()}\n"


class MemoryStorage:
    """
    记忆物理存储层与索引管理器
    """

    INDEX_FILENAME = "MEMORY.md"

    def __init__(self, workdir: Path, memory_dir: Optional[Path] = None):
        self.workdir = Path(workdir).resolve()
        self.memory_dir = (memory_dir or (self.workdir / ".memory")).resolve()
        self._lock = asyncio.Lock()

    @staticmethod
    def _atomic_write_text(path: Path, content: str) -> None:
        """
        POSIX 标准原子写入：
        先写入同目录下的隐式临时文件，再通过 os.replace 进行原子重命名替换。
        彻底规避进程崩溃导致的文件截断损坏，以及并发读取到未完成的半截内容。
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex[:6]}.tmp")
        try:
            tmp_path.write_text(content, encoding="utf-8")
            os.replace(tmp_path, path)
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

    def resolve_path(self, filename: str, allow_index: bool = False) -> Path:
        """
        防逃逸路径解析：
        1. 文件名不能包含路径分隔符；
        2. 默认禁止直接访问 MEMORY.md 索引文件（索引由存储层统一维护）；
        3. 目标路径必须严格位于 memory_dir 内部，且 memory_dir 必须在 workdir 内部。
        """
        if Path(filename).name != filename:
            raise ValueError(f"非法记忆文件名（禁止包含路径层级）: {filename}")
        if filename == self.INDEX_FILENAME and not allow_index:
            raise ValueError("MEMORY.md 为系统专用索引文件，不能作为普通记忆记录直接读写")

        if not self.memory_dir.is_relative_to(self.workdir):
            raise ValueError(f"记忆目录逃逸出当前工作区: {self.memory_dir}")

        path = (self.memory_dir / filename).resolve()
        if not path.is_relative_to(self.memory_dir):
            raise ValueError(f"记忆文件路径逃逸出记忆存储目录: {filename}")
        return path

    def write_memory(self, name: str, mem_type: MemoryType | str, description: str, body: str) -> Path:
        """
        写入或更新单条记忆文件，支持 Slug 冲突智能消歧，并通过原子写与索引重建完成持久化
        """
        name = name.strip()
        description = description.strip()
        body = body.strip()

        if not name:
            raise ValueError("记忆名称 (name) 不能为空")
        if not description or not body:
            raise ValueError("记忆描述 (description) 与正文 (body) 不能为空")

        type_enum = mem_type if isinstance(mem_type, MemoryType) else MemoryType.from_str(str(mem_type))

        self.memory_dir.mkdir(parents=True, exist_ok=True)
        base_slug = memory_slug(name)
        filename = f"{base_slug}.md"
        path = self.resolve_path(filename)

        # 智能冲突消歧：若文件已存在，检查已有文件的真实标题
        if path.is_file():
            try:
                existing_meta, _ = parse_frontmatter(path.read_text(encoding="utf-8"))
                existing_name = str(existing_meta.get("name") or "").strip()
                # 若名称不一致（如 "My Mind" 与 "my-mind"），自动追加自增后缀防覆盖
                if existing_name and existing_name.lower() != name.lower():
                    counter = 2
                    while True:
                        candidate_name = f"{base_slug}-{counter}.md"
                        candidate_path = self.resolve_path(candidate_name)
                        if not candidate_path.is_file():
                            filename = candidate_name
                            path = candidate_path
                            break
                        # 如果已存在的 candidate 正好就是当前想要更新的 name，则命中复用
                        c_meta, _ = parse_frontmatter(candidate_path.read_text(encoding="utf-8"))
                        if str(c_meta.get("name") or "").strip().lower() == name.lower():
                            filename = candidate_name
                            path = candidate_path
                            break
                        counter += 1
            except Exception as e:
                logger.debug(f"[Memory Storage] 读取现有记忆检测冲突失败，继续执行写入: {e}")

        content = format_document(name, type_enum.value, description, body)
        self._atomic_write_text(path, content)

        self.rebuild_index()
        return path

    def read_memory(self, filename: str) -> Optional[str]:
        """
        读取单条记忆文件的原始全文
        """
        try:
            path = self.resolve_path(filename)
        except ValueError:
            return None
        if path.is_file():
            try:
                return path.read_text(encoding="utf-8")
            except Exception as e:
                logger.warning(f"[Memory Storage] 读取记忆文件失败: {filename} | {e}")
                return None
        return None

    def delete_memory(self, filename: str) -> bool:
        """
        删除单条记忆文件并重建索引
        """
        try:
            path = self.resolve_path(filename)
        except ValueError:
            return False
        if path.is_file():
            try:
                path.unlink()
                self.rebuild_index()
                return True
            except Exception as e:
                logger.warning(f"[Memory Storage] 删除记忆文件失败: {filename} | {e}")
                return False
        return False

    def list_memories(self) -> List[MemoryRecord]:
        """
        加载全部合法的长期记忆实体列表（内置单文件损坏容错）
        """
        records: List[MemoryRecord] = []
        if not self.memory_dir.exists():
            return records

        for path in sorted(self.memory_dir.glob("*.md")):
            if path.name == self.INDEX_FILENAME:
                continue
            try:
                safe_path = self.resolve_path(path.name)
            except ValueError:
                continue

            try:
                content = safe_path.read_text(encoding="utf-8")
                metadata, body = parse_frontmatter(content)
                name = str(metadata.get("name") or path.stem).strip()
                mem_type = MemoryType.from_str(str(metadata.get("type") or "project"))
                description = str(metadata.get("description") or "").strip()

                records.append(
                    MemoryRecord(
                        name=name,
                        type=mem_type,
                        description=description,
                        body=body.strip(),
                        filename=path.name,
                    )
                )
            except Exception as e:
                logger.warning(f"[Memory Storage] 解析单条记忆文件失败，已自动跳过: {path.name} | {e}")
                continue

        return records

    def rebuild_index(self) -> None:
        """
        从所有单条记忆文件中提取摘要，重新生成 MEMORY.md 索引目录（原子落盘）
        """
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        lines = []
        for path in sorted(self.memory_dir.glob("*.md")):
            if path.name == self.INDEX_FILENAME:
                continue
            try:
                safe_path = self.resolve_path(path.name)
            except ValueError:
                continue

            try:
                metadata, body = parse_frontmatter(safe_path.read_text(encoding="utf-8"))
                name = " ".join(str(metadata.get("name") or safe_path.stem).split())
                first_line = next((line for line in body.splitlines() if line.strip()), "")
                description = " ".join(str(metadata.get("description") or first_line).split())
                lines.append(f"- [{name}]({safe_path.name}) - {description}")
            except Exception as e:
                logger.warning(f"[Memory Storage] 索引解析跳过损坏文件: {path.name} | {e}")
                continue

        index_path = self.resolve_path(self.INDEX_FILENAME, allow_index=True)
        index_content = "\n".join(lines) + ("\n" if lines else "")
        self._atomic_write_text(index_path, index_content)

    def read_index(self) -> str:
        """
        读取 MEMORY.md 索引全文
        """
        try:
            path = self.resolve_path(self.INDEX_FILENAME, allow_index=True)
        except ValueError:
            return ""
        return path.read_text(encoding="utf-8").strip() if path.exists() else ""

    def create_snapshot(self) -> Dict[str, str]:
        """
        创建当前全部记忆文件的内存快照（用于整理失败时的原子回滚）
        """
        snapshot: Dict[str, str] = {}
        if not self.memory_dir.exists():
            return snapshot

        for path in self.memory_dir.glob("*.md"):
            if path.name == self.INDEX_FILENAME:
                continue
            try:
                safe_path = self.resolve_path(path.name)
                snapshot[safe_path.name] = safe_path.read_text(encoding="utf-8")
            except Exception:
                continue
        return snapshot

    def restore_snapshot(self, snapshot: Dict[str, str]) -> None:
        """
        将记忆存储完全还原为快照状态，并通过原子写重建索引
        """
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        # 清除当前可能已部分写入的不一致文件
        for path in self.memory_dir.glob("*.md"):
            if path.name != self.INDEX_FILENAME:
                try:
                    self.resolve_path(path.name).unlink()
                except (ValueError, OSError):
                    continue

        # 还原快照中的文件
        for filename, content in snapshot.items():
            try:
                safe_path = self.resolve_path(filename)
                self._atomic_write_text(safe_path, content)
            except Exception as e:
                logger.error(f"[Memory Storage] 快照还原单文件失败: {filename} | {e}")

        self.rebuild_index()
