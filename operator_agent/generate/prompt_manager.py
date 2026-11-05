"""
Prompt 管理器
从 prompts/ 目录加载 .md 模板，支持 {{variable}} 占位符渲染
集成 skills/ 目录的 SKILL 规范，支持按 Agent 角色切片注入
"""

import re
from pathlib import Path
from typing import Optional

from ..generate.generation_spec import GENERATION_SPEC


# SKILL.md 中各 section 对应的 Agent 角色映射
# key = section 编号, value = 该 section 注入给哪些 Agent
_SKILL_SECTION_AGENTS = {
    "1": ["coder"],       # Prohibitions -> Coder (生成代码时必须遵守)
    "2": ["coder"],       # Required Patterns -> Coder
    "3": ["search"],      # Optimization Priority -> Search (优化时参考)
    "4": ["designer"],    # Tile Size Guidelines -> Designer (设计 tiling)
    "5": ["designer", "coder"],  # Common Patterns -> Designer + Coder
    "6": ["coder"],       # Correctness Checklist -> Coder
}


class PromptManager:
    """
    Prompt 模板管理器 + Skill 规范注入器

    - 从 prompts/ 目录加载 .md 模板
    - 从 skills/ 目录加载 SKILL 规范
    - render() 时自动注入对应 Agent 角色的 SKILL 片段
    - 支持 {{variable}} 占位符渲染
    - 支持 {{skill:<section>}} 语法在模板中显式引用 SKILL 片段

    使用方式:
        pm = PromptManager()
        prompt = pm.render("coder_generate_triton", kernel_type="matmul",
                          _agent_role="coder", block_m=64, ...)
    """

    def __init__(
        self,
        prompts_dir: Optional[str] = None,
        skills_dir: Optional[str] = None,
        skill_file: Optional[str] = None,
        generation_instructions: str = "",
    ):
        base = Path(__file__).resolve().parents[1]
        self._prompts_dir = Path(prompts_dir) if prompts_dir else base / "prompts"
        self._skills_dir = Path(skills_dir) if skills_dir else base / "skills"
        self._skill_file = skill_file
        self._generation_instructions = generation_instructions
        self._template_cache: dict[str, str] = {}
        self._skill_cache: Optional[dict[str, str]] = None

    def list_prompts(self) -> list[str]:
        if not self._prompts_dir.exists():
            return []
        return [f.stem for f in self._prompts_dir.glob("*.md")]

    def list_skills(self) -> list[str]:
        if not self._skills_dir.exists():
            return []
        return [f.name for f in self._skills_dir.glob("*.md")]

    # ------------------------------------------------------------------
    # Skill 加载
    # ------------------------------------------------------------------

    def _load_skill(self, skill_name: str) -> str:
        """加载并缓存完整 SKILL 文件"""
        path = self._skills_dir / f"{skill_name}.md"
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8").strip()

    def _parse_skill_sections(self, skill_content: str) -> dict[str, str]:
        """
        将 SKILL 文件按 ## N. 标题 切片为 {section_id: section_content}
        返回的 dict 可按 section 编号索引
        """
        sections: dict[str, str] = {}
        current_num = ""
        current_lines: list[str] = []

        for line in skill_content.split("\n"):
            # 匹配 "## 1. xxx" 或 "## 1 xxx" 格式
            m = re.match(r"^##\s+(\d+)\s*[.．]\s*", line)
            if m:
                if current_num and current_lines:
                    sections[current_num] = "\n".join(current_lines).strip()
                current_num = m.group(1)
                current_lines = [line]
            elif current_num:
                current_lines.append(line)

        if current_num and current_lines:
            sections[current_num] = "\n".join(current_lines).strip()

        return sections

    def _get_skill_for_agent(self, agent_role: str, skill_name: str) -> str:
        """
        根据 Agent 角色筛选出相关的 SKILL 片段，拼接为注入文本。
        agent_role: "designer" / "coder" / "search"
        """
        raw = self._load_skill(skill_name)
        if not raw:
            return ""

        sections = self._parse_skill_sections(raw)
        if not sections:
            return raw

        parts = []
        for sec_id, sec_content in sections.items():
            allowed_agents = _SKILL_SECTION_AGENTS.get(sec_id, [])
            if agent_role.lower() in allowed_agents:
                parts.append(sec_content)

        if not parts:
            return ""

        header = f"> SKILL SPECIFICATION ({agent_role.upper()} role)"
        return header + "\n" + "\n---\n".join(parts)

    def get_skill_section(self, section_num: str, skill_name: Optional[str] = None) -> str:
        """获取指定 section 编号的 SKILL 片段（供模板 {{skill:N}} 引用）"""
        name = skill_name or self._skill_file
        if not name:
            return ""
        raw = self._load_skill(name)
        if not raw:
            return ""
        sections = self._parse_skill_sections(raw)
        return sections.get(section_num, "")

    # ------------------------------------------------------------------
    # Template 加载与渲染
    # ------------------------------------------------------------------

    def get_raw(self, name: str) -> str:
        if name in self._template_cache:
            return self._template_cache[name]
        path = self._prompts_dir / f"{name}.md"
        if not path.exists():
            raise FileNotFoundError(f"Prompt template not found: {path}")
        content = path.read_text(encoding="utf-8").strip()
        self._template_cache[name] = content
        return content

    def render(self, name: str, **kwargs) -> str:
        """
        渲染模板，替换所有 {{key}} 为 kwargs 中的值。
        支持两种特殊占位符:
          - {{skill}}: 注入当前 Agent 角色的完整 SKILL 片段
          - {{skill:N}}: 注入 SKILL 的第 N 个 section
        需要 kwargs 中传入 _agent_role (designer/coder/search) 和
        _skill_name (如 "triton_skill")，否则不注入 SKILL。
        """
        template = self.get_raw(name)
        agent_role = kwargs.pop("_agent_role", None)
        skill_name = kwargs.pop("_skill_name", self._skill_file)

        def replacer(match: re.Match) -> str:
            key = match.group(1).strip()

            # 处理 {{skill}} 和 {{skill:N}} 语法
            if key == "skill":
                if agent_role and skill_name:
                    return self._get_skill_for_agent(agent_role, skill_name)
                return ""
            if key.startswith("skill:"):
                sec_num = key.split(":", 1)[1].strip()
                return self.get_skill_section(sec_num, skill_name)

            # 普通变量替换
            if key not in kwargs:
                return match.group(0)
            val = kwargs[key]
            return str(val) if val is not None else ""

        rendered = re.sub(r"\{\{(.+?)\}\}", replacer, template)
        # Template names also cover callers which do not pass _agent_role
        # (notably Fixer); no new placeholder or per-call opt-in is required.
        if (name.startswith(("designer_", "planner_", "coder_", "fixer_", "search_"))
                or agent_role in ("designer", "planner", "coder", "fixer", "search")):
            prefix = GENERATION_SPEC
            if self._generation_instructions:
                prefix += "\n\n" + self._generation_instructions
            return prefix + "\n\n" + rendered
        return rendered

    def reload(self, name: Optional[str] = None):
        if name:
            self._template_cache.pop(name, None)
        else:
            self._template_cache.clear()
        self._skill_cache = None

    def __repr__(self) -> str:
        prompts = self.list_prompts()
        skills = self.list_skills()
        return (f"PromptManager(templates={len(prompts)}, "
                f"skills={len(skills)}, dir={self._prompts_dir})")
