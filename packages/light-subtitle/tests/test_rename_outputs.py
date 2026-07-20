"""Bare-name output layout — rename helpers were removed."""

from __future__ import annotations


def test_rename_outputs_helpers_removed() -> None:
    """CLI no longer slug-prefixes exports; helpers must stay gone."""
    import light_subtitle.cli as cli

    assert not hasattr(cli, "_rename_outputs")
    assert not hasattr(cli, "_has_generic_outputs")
