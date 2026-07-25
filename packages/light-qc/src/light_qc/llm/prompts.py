"""Prompt rendering for the LLM QC pass.

The bundled ``qc.j2`` template lives under ``light_qc/prompts/`` and is
loaded via :mod:`importlib.resources`, keeping the package self-contained
(same pattern as ``light_asr_polish.prompts`` / ``light_subtitle.prompts``).
"""

from importlib.resources import files
from pathlib import Path

from jinja2 import BaseLoader, Environment, FileSystemLoader


def render(template_path: str, **kwargs) -> str:
    path = Path(template_path)
    if path.exists():
        loader = FileSystemLoader(path.parent)
        env = Environment(loader=loader)
        template = env.get_template(path.name)
    else:
        env = Environment(loader=BaseLoader())
        template = env.from_string(template_path)
    return template.render(**kwargs)


def render_prompt(name: str, **kwargs) -> str:
    """Render a bundled ``.j2`` template from ``light_qc/prompts/``."""
    template = (files("light_qc") / "prompts" / name).read_text(encoding="utf-8")
    env = Environment(loader=BaseLoader())
    return env.from_string(template).render(**kwargs)
