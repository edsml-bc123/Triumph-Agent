from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union

import yaml


@dataclass
class SkillSpec:
    """技能定义规范"""
    name: str
    description: str
    path: Path
    content: str
    body: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class SkillLoader:
    """
    技能动态发现与按需加载器 (Progressive Disclosure)
    
    1. 扫描指定目录下的所有 */SKILL.md 文件；
    2. 严格执行路径越界防护 (Path Traversal Guard)；
    3. 仅提取元数据 (name, description) 用于 System Prompt 中的极简 Catalog，杜绝 Token 膨胀；
    4. 提供按需加载接口，仅在模型主动调用时返回完整指令与操作规范。
    """

    def __init__(self, skills_dir: Union[Path, str, Sequence[Union[Path, str]]]):
        if isinstance(skills_dir, (str, Path)):
            self.skills_dirs: List[Path] = [Path(skills_dir)]
        else:
            self.skills_dirs = [Path(d) for d in skills_dir]

        self.skills: Dict[str, SkillSpec] = {}
        self.scan()

    @staticmethod
    def parse_frontmatter(text: str) -> tuple[Dict[str, Any], str]:
        """
        解析 Markdown 开头的 YAML Frontmatter 区域。
        格式形如：
        ---
        name: my-skill
        description: A useful skill
        ---
        # Skill Body
        """
        lines = text.splitlines(keepends=True)
        if not lines or lines[0].rstrip("\r\n") != "---":
            return {}, text

        closing_index = next(
            (idx for idx, line in enumerate(lines[1:], start=1) if line.rstrip("\r\n") == "---"),
            None,
        )
        if closing_index is None:
            return {}, text

        frontmatter_str = "".join(lines[1:closing_index])
        body = "".join(lines[closing_index + 1:]).strip()

        try:
            metadata = yaml.safe_load(frontmatter_str) or {}
        except yaml.YAMLError:
            metadata = {}

        if not isinstance(metadata, dict):
            metadata = {}

        return metadata, body

    def scan(self) -> Dict[str, SkillSpec]:
        """
        执行技能目录地毯式扫描，加载所有合法的 SKILL.md。
        防范目录逃逸软链接风险。
        """
        self.skills.clear()

        for directory in self.skills_dirs:
            if not directory.exists():
                continue

            root_resolved = directory.resolve()
            for manifest in sorted(directory.glob("*/SKILL.md")):
                manifest_resolved = manifest.resolve()
                # 路径防越权沙箱校验：必须严格处于合法根目录下
                try:
                    if not manifest_resolved.is_relative_to(root_resolved):
                        continue
                except (ValueError, AttributeError):
                    if not str(manifest_resolved).startswith(str(root_resolved)):
                        continue

                if not manifest_resolved.is_file():
                    continue

                try:
                    content = manifest_resolved.read_text(encoding="utf-8")
                except Exception:
                    continue

                metadata, body = self.parse_frontmatter(content)

                # 提取名称
                raw_name = metadata.get("name")
                name = raw_name.strip() if isinstance(raw_name, str) else ""
                if not name:
                    name = manifest.parent.name

                # 提取描述：优先 metadata，其次使用 body 首行非空文字
                raw_desc = metadata.get("description")
                desc = raw_desc.strip() if isinstance(raw_desc, str) else ""
                if not desc and body:
                    first_line = next((ln.strip() for ln in body.splitlines() if ln.strip()), "")
                    desc = re.sub(r"^#+\s*", "", first_line).strip()

                desc = " ".join(desc.split()) if desc else "无描述"

                self.skills[name] = SkillSpec(
                    name=name,
                    description=desc,
                    path=manifest_resolved,
                    content=content,
                    body=body,
                    metadata=metadata,
                )

        return self.skills

    def catalog(self) -> str:
        """
        输出紧凑的技能清单，专用于挂载至 System Prompt。
        格式：
        - {name}: {description}
        """
        if not self.skills:
            return "(no skills found)"
        return "\n".join(
            f"- {skill.name}: {skill.description}"
            for skill in sorted(self.skills.values(), key=lambda s: s.name)
        )

    def load(self, name: str) -> str:
        """
        根据技能名称按需加载完整内容。
        若不存在则返回明确的错误回显及当前可用技能候选。
        """
        skill = self.skills.get(name.strip())
        if skill:
            return skill.content

        available = ", ".join(sorted(self.skills.keys())) or "none"
        return f"Error: Unknown skill '{name}'. Available skills: {available}"


class SkillTool:
    """
    独立技能加载工具 (符合 ToolPlugin 协议契约)
    提供标准 load_skill 工具向 ToolRegistry 注册。
    """

    def __init__(self, loader: SkillLoader):
        self.loader = loader

    def register_to(self, registry: Any) -> None:
        """向指定的 ToolRegistry 挂载 load_skill 工具"""
        registry.register(
            name="load_skill",
            description="按技能名称加载技能的完整执行指导规范与操作流程 (SKILL.md)",
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "要加载的技能名称（从系统提示词的可用技能列表中选择）",
                    }
                },
                "required": ["name"],
            },
            handler=self.loader.load,
        )

