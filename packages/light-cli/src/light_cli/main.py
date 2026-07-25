"""Light — entry points.

``light-subtitle`` (``app``): the full pipeline CLI, defined in cli.py.
``light`` (``light_app``): multi-command toolbox — ``pipeline`` reuses the
very same callback (identical flags and behavior), while ``asr`` /
``polish`` / ``subtitle`` are standalone capability commands over
light-asr / light-asr-polish / light-subtitle (see :mod:`light_cli.commands`).
"""

import typer

from .cli import app, run
from .commands import asr as asr_command
from .commands import polish as polish_command
from .commands import subtitle as subtitle_command


def main():
    app()


light_app = typer.Typer(
    help="Light — 视频/音频 → 字幕工具箱：pipeline 全流程，或 asr/polish/subtitle 独立步骤。",
    no_args_is_help=True,
)
# Same function registered as the light-subtitle callback: identical flags,
# identical behavior — no parameter duplication.
light_app.command("pipeline")(run)
light_app.command("asr")(asr_command.asr)
light_app.command("polish")(polish_command.polish)
light_app.command("subtitle")(subtitle_command.subtitle)
