import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from client import DashScopeClient
from runtime.hooks import HookManager
from runtime.loop import AgentLoop
from skills.loader import SkillLoader, SkillTool
from tools.builtin import BuiltinTool
from tools.registry import ToolRegistry


class TestSkillLoader:
    def test_parse_frontmatter_valid(self):
        text = """---
name: code-review
description: 深度代码审查与坏味道排查规范
tags: [python, review]
---
# 代码审查指南
请检查并发与死锁问题。
"""
        metadata, body = SkillLoader.parse_frontmatter(text)
        assert metadata["name"] == "code-review"
        assert metadata["description"] == "深度代码审查与坏味道排查规范"
        assert metadata["tags"] == ["python", "review"]
        assert body == "# 代码审查指南\n请检查并发与死锁问题。"

    def test_parse_frontmatter_no_frontmatter(self):
        text = "# 纯 Markdown\n无 Frontmatter。"
        metadata, body = SkillLoader.parse_frontmatter(text)
        assert metadata == {}
        assert body == text

    def test_parse_frontmatter_unclosed(self):
        text = "---\nname: unclosed\n没有闭合的三道杠"
        metadata, body = SkillLoader.parse_frontmatter(text)
        assert metadata == {}
        assert body == text

    def test_parse_frontmatter_invalid_yaml(self):
        text = "---\n: : : 非法 yaml\n---\n正文"
        metadata, body = SkillLoader.parse_frontmatter(text)
        assert metadata == {}
        assert body == "正文"

    def test_scan_and_fallback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            skills_dir = Path(tmpdir) / "skills"
            skills_dir.mkdir()

            # 技能 1：带完整 Frontmatter
            s1_dir = skills_dir / "git-release"
            s1_dir.mkdir()
            (s1_dir / "SKILL.md").write_text(
                "---\nname: git-release\ndescription: 标准发版流程\n---\n# 发版正文",
                encoding="utf-8",
            )

            # 技能 2：无 Frontmatter，使用目录名和首行作为 fallback
            s2_dir = skills_dir / "perf-audit"
            s2_dir.mkdir()
            (s2_dir / "SKILL.md").write_text(
                "# 性能审计深度分析\n详细排查慢查询与瓶颈点。",
                encoding="utf-8",
            )

            loader = SkillLoader(skills_dir)
            skills = loader.skills

            assert "git-release" in skills
            assert skills["git-release"].description == "标准发版流程"

            assert "perf-audit" in skills
            assert skills["perf-audit"].description == "性能审计深度分析"

            catalog = loader.catalog()
            assert "- git-release: 标准发版流程" in catalog
            assert "- perf-audit: 性能审计深度分析" in catalog

    def test_path_traversal_guard(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            skills_dir = base_dir / "skills"
            outside_dir = base_dir / "outside"
            skills_dir.mkdir()
            outside_dir.mkdir()

            # 在外部目录写一个 SKILL.md
            outside_skill = outside_dir / "SKILL.md"
            outside_skill.write_text("---\nname: evil\n---\nEvil Content", encoding="utf-8")

            # 在 skills 目录下软链接一个指向外部的子目录
            symlink_dir = skills_dir / "evil_link"
            try:
                symlink_dir.symlink_to(outside_dir, target_is_directory=True)
            except (OSError, NotImplementedError):
                pytest.skip("当前操作系统环境不支持符号链接测试")

            loader = SkillLoader(skills_dir)
            # 外部软链接应当被 is_relative_to 安全沙箱拦截，不被收录
            assert "evil" not in loader.skills

    def test_load_skill_success_and_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            skills_dir = Path(tmpdir) / "skills"
            skills_dir.mkdir()

            s_dir = skills_dir / "test-skill"
            s_dir.mkdir()
            (s_dir / "SKILL.md").write_text(
                "---\nname: test-skill\ndescription: 测试专用技能\n---\n# 执行步骤\n1. 验证单测",
                encoding="utf-8",
            )

            loader = SkillLoader(skills_dir)

            content = loader.load("test-skill")
            assert "# 执行步骤" in content
            assert "1. 验证单测" in content

            # 加载不存在的技能
            error_msg = loader.load("non-exist")
            assert "Error: Unknown skill 'non-exist'" in error_msg
            assert "test-skill" in error_msg


class TestSkillsIntegration:
    async def test_skills_plugin_registration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            skills_dir = Path(tmpdir) / "skills"
            skills_dir.mkdir()

            s_dir = skills_dir / "deploy"
            s_dir.mkdir()
            (s_dir / "SKILL.md").write_text(
                "---\nname: deploy\ndescription: 线上发布部署规范\n---\n执行部署步骤",
                encoding="utf-8",
            )

            loader = SkillLoader(skills_dir)
            registry = ToolRegistry(workdir=skills_dir)

            plugin = SkillTool(loader)
            plugin.register_to(registry)

            # 验证工具已注册
            tool_names = {t["function"]["name"] for t in registry.get_tools_spec()}
            assert "load_skill" in tool_names
            assert "deploy" in loader.skills

            # 模拟执行
            result = await registry.execute("load_skill", {"name": "deploy"})
            assert "执行部署步骤" in result

    async def test_plugin_composition_with_builtin_tools(self):
        """验证标准工具插件组合机制：Builtin 与 Skills 独立插件并存无冲突"""
        with tempfile.TemporaryDirectory() as tmpdir:
            skills_dir = Path(tmpdir) / "skills"
            skills_dir.mkdir()
            s_dir = skills_dir / "clean"
            s_dir.mkdir()
            (s_dir / "SKILL.md").write_text(
                "---\nname: clean\ndescription: 代码清理规范\n---\n清理所有临时缓存",
                encoding="utf-8",
            )

            loader = SkillLoader(skills_dir)
            registry = ToolRegistry(workdir=skills_dir)

            registry.register_plugin(BuiltinTool(workdir=skills_dir))
            registry.register_plugin(SkillTool(loader))

            tool_names = {t["function"]["name"] for t in registry.get_tools_spec()}
            expected = {"bash", "read_file", "write_file", "edit_file", "glob", "load_skill"}
            assert expected.issubset(tool_names)

            res = await registry.execute("load_skill", {"name": "clean"})
            assert "清理所有临时缓存" in res

    def test_agent_loop_dynamic_system_prompt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            skills_dir = Path(tmpdir) / "skills"
            skills_dir.mkdir()
            s_dir = skills_dir / "security-check"
            s_dir.mkdir()
            (s_dir / "SKILL.md").write_text(
                "---\nname: security-check\ndescription: 零信任安全检查指引\n---\n执行漏洞扫描",
                encoding="utf-8",
            )

            loader = SkillLoader(skills_dir)
            registry = ToolRegistry(workdir=skills_dir)
            mock_client = MagicMock(spec=DashScopeClient)
            hooks = HookManager()

            loop = AgentLoop(
                registry=registry,
                client=mock_client,
                hooks=hooks,
                skill_loader=loader,
            )

            prompt = loop._default_system_prompt()
            assert "可用专业技能库 (Skills):" in prompt
            assert "- security-check: 零信任安全检查指引" in prompt
            assert "load_skill(name=...)" in prompt

    async def test_agent_loop_e2e_load_skill(self):
        from unittest.mock import AsyncMock
        from client import LLMResponse, ToolCall
        from runtime.state import AgentState, AgentStatus

        with tempfile.TemporaryDirectory() as tmpdir:
            skills_dir = Path(tmpdir) / "skills"
            skills_dir.mkdir()
            s_dir = skills_dir / "code-review"
            s_dir.mkdir()
            (s_dir / "SKILL.md").write_text(
                "---\nname: code-review\ndescription: 代码审查指南\n---\n# 规范：必须检查死锁与竞态",
                encoding="utf-8",
            )

            loader = SkillLoader(skills_dir)
            registry = ToolRegistry(workdir=skills_dir)
            registry.register_plugin(BuiltinTool(workdir=skills_dir))
            registry.register_plugin(SkillTool(loader))

            mock_client = MagicMock(spec=DashScopeClient)
            mock_client.default_model = "qwen3.8-flash"
            mock_client.evaluator_model = "qwen3.6-flash"
            mock_client.chat_completion = AsyncMock()

            # 模拟第 1 轮：模型调用 load_skill；第 2 轮：模型根据技能规范给出最终审查结论
            mock_client.chat_completion.side_effect = [
                LLMResponse(
                    content="我先加载 code-review 技能规范。",
                    finish_reason="tool_calls",
                    tool_calls=[
                        ToolCall(
                            id="call_skill_1",
                            name="load_skill",
                            arguments_raw='{"name": "code-review"}',
                        )
                    ],
                ),
                LLMResponse(
                    content="代码审查完成：经检查未发现死锁与竞态问题，符合规范。",
                    finish_reason="stop",
                    tool_calls=[],
                ),
            ]

            hooks = HookManager()
            loop = AgentLoop(
                registry=registry,
                client=mock_client,
                hooks=hooks,
                skill_loader=loader,
            )

            state = AgentState()
            state.add_user_message("帮我审查一下这段代码。")

            final_state = await loop.run(state)

            assert final_state.status == AgentStatus.SUCCESS
            assert final_state.step_count == 2
            assert "经检查未发现死锁与竞态问题" in final_state.final_answer
            # 验证工具调用回填消息中包含了技能文档内容
            tool_msg = next(m for m in final_state.messages if m.get("role") == "tool")
            assert "必须检查死锁与竞态" in tool_msg["content"]

