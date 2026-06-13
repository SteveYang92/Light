"""Light Subtitle — entry point.

The CLI layer (arguments, config, dispatch) lives in cli.py.
Pipeline orchestration lives in orchestrator.py.
This module exists only for the ``light-subtitle`` console_scripts entry point.
"""

from .cli import app


def main():
    app()
