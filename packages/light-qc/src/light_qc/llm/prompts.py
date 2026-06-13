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
    project_root = Path(__file__).parent.parent.parent.parent.parent.parent
    prompt_dir = project_root / "prompts"
    path = prompt_dir / name
    if path.exists():
        return render(str(path), **kwargs)
    return ""
